import streamlit as st
import pandas as pd
import numpy as np
import json
import faiss
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as train_test_split_surprise
import shap
from lime import lime_tabular
import os

st.set_page_config(layout="wide", page_title="Music Recommender System")

# --- Paths ---
save_dir = r'Streamlit_Frontend'

CLEANED_DATA_PATH             = os.path.join(save_dir, 'cleaned_data.csv')
FEATURE_ENGINEERED_DATA_PATH  = os.path.join(save_dir, 'feature_engineered_data.csv')
COSINE_SIM_PATH               = os.path.join(save_dir, 'cosine_similarity_matrix.npy')
SYNTHETIC_RATINGS_PATH        = os.path.join(save_dir, 'synthetic_ratings.csv')
DL_MODEL_PATH                 = os.path.join(save_dir, 'deep_learning_model.keras')
FAISS_INDEX_PATH              = os.path.join(save_dir, 'faiss_index.bin')
EVALUATION_RESULTS_PATH       = os.path.join(save_dir, 'evaluation_results.json')
SENTENCE_TRANSFORMER_MODEL_NAME = 'all-MiniLM-L6-v2'


# =============================================================================
# FIX 1: PatchedEmbedding — tolerates unknown kwargs (e.g. quantization_config)
#         introduced in newer Keras versions.
# =============================================================================
class PatchedEmbedding(tf.keras.layers.Embedding):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)



def _strip_unknown_keys(obj):
    """Recursively remove config keys older Keras versions don't recognise."""
    UNKNOWN_KEYS = {'quantization_config'}
    if isinstance(obj, dict):
        for key in UNKNOWN_KEYS:
            obj.pop(key, None)
        for v in obj.values():
            _strip_unknown_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_unknown_keys(item)


def load_dl_model(path: str):
    """
    Load a .keras model tolerating Keras version mismatches.

    Attempt 1 - plain load (works when versions match).
    Attempt 2 - patch the Keras global deserialisation registry so every
                Embedding layer is rebuilt via PatchedEmbedding.
    Attempt 3 - rewrite the .keras ZIP with a sanitised config.json,
                then load that patched copy.
    """
    import zipfile, shutil, tempfile

    # Attempt 1: plain
    try:
        return tf.keras.models.load_model(path)
    except Exception as e1:
        first_error = e1

    # Attempt 2: patch Keras global registry
    try:
        from keras.src.saving import serialization_lib
        registry = serialization_lib._CONFIG_REGISTRY
        original = registry.get('keras>Embedding')
        registry['keras>Embedding'] = PatchedEmbedding
        try:
            return tf.keras.models.load_model(path)
        finally:
            if original is not None:
                registry['keras>Embedding'] = original
            else:
                registry.pop('keras>Embedding', None)
    except Exception as e2:
        second_error = e2

    # Attempt 3: sanitise config.json inside the ZIP, load from patched copy
    try:
        tmp_dir   = tempfile.mkdtemp()
        tmp_keras = os.path.join(tmp_dir, 'model_patched.keras')

        with zipfile.ZipFile(path, 'r') as zin:
            all_files = {name: zin.read(name) for name in zin.namelist()}

        raw_config = json.loads(all_files['config.json'])
        _strip_unknown_keys(raw_config)
        all_files['config.json'] = json.dumps(raw_config).encode('utf-8')

        with zipfile.ZipFile(tmp_keras, 'w', zipfile.ZIP_DEFLATED) as zout:
            for name, data in all_files.items():
                zout.writestr(name, data)

        try:
            return tf.keras.models.load_model(
                tmp_keras,
                custom_objects={'Embedding': PatchedEmbedding}
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    except Exception as e3:
        raise RuntimeError(
            f"All three attempts to load '{path}' failed.\n"
            f"  Attempt 1 (plain load):          {first_error}\n"
            f"  Attempt 2 (registry patch):      {second_error}\n"
            f"  Attempt 3 (sanitise ZIP config): {e3}\n\n"
            "Permanent fix: open your training notebook and re-save:\n"
            "  import tensorflow as tf\n"
            "  m = tf.keras.models.load_model('deep_learning_model.keras')\n"
            "  m.save('deep_learning_model.keras')"
        )


# =============================================================================
# Cached loader — returns every artifact as a plain dict so callers can use
# normal local variables (avoids the UnboundLocalError on dl_model).
# =============================================================================
@st.cache_resource
def load_all_artifacts():
    # ---- DataFrames ----
    df               = pd.read_csv(CLEANED_DATA_PATH)
    df_fe            = pd.read_csv(FEATURE_ENGINEERED_DATA_PATH)
    cosine_sim       = np.load(COSINE_SIM_PATH)
    synthetic_ratings_df = pd.read_csv(SYNTHETIC_RATINGS_PATH)

    # ---- Deep Learning model ----
    # Initialise to None BEFORE the try block so Python's compiler never
    # treats dl_model as an "unassigned local" further down in this function.
    dl_model = None
    try:
        dl_model = load_dl_model(DL_MODEL_PATH)
    except Exception as e:
        st.error(
            f"Failed to load Deep Learning model.\n\n**Error:** {e}\n\n"
            "**Tip:** Re-save the model in your training notebook with your "
            "current TF/Keras version: `model.save('deep_learning_model.keras')`"
        )

    # Hard stop here — st.stop() alone is not guaranteed to prevent execution
    # from continuing in all Streamlit versions, so we raise explicitly.
    if dl_model is None:
        raise RuntimeError(
            "Deep Learning model could not be loaded. "
            "Check the error message above and re-save the model."
        )

    # ---- FAISS ----
    embedding_cols = df_fe.filter(regex='song_name_embedding_')
    if embedding_cols.empty:
        st.error("Cannot find 'song_name_embedding_' columns in feature-engineered data.")
        st.stop()
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    # ---- Sentence Transformer ----
    st_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)

    # ---- Evaluation results ----
    with open(EVALUATION_RESULTS_PATH, 'r') as f:
        eval_results = json.load(f)

    # ---- Mappings ----
    song_to_item_id  = {name: i for i, name in enumerate(df['Song-Name'].unique())}
    item_id_to_song  = {i: name for name, i in song_to_item_id.items()}

    unique_user_ids  = synthetic_ratings_df['user_id'].unique()
    user_id_to_idx   = {uid: idx for idx, uid in enumerate(unique_user_ids)}
    idx_to_user_id   = {idx: uid for uid, idx in user_id_to_idx.items()}

    num_users = len(unique_user_ids)
    num_items = len(df['Song-Name'].unique())

    # ---- Surprise SVD ----
    reader   = Reader(rating_scale=(1, 10))
    data     = Dataset.load_from_df(
        synthetic_ratings_df[['user_id', 'item_id', 'rating']], reader
    )
    trainset, _ = train_test_split_surprise(data, test_size=0.001, random_state=42)
    algo_svd = SVD()
    algo_svd.fit(trainset)

    # ---- Explainer model (Dense-only, for SHAP / LIME) ----
    embedding_dim = 50  # must match training script

    expl_input  = tf.keras.Input(shape=(embedding_dim * 2,))
    x           = tf.keras.layers.Dense(128, activation='relu', name='dense_expl_1')(expl_input)
    x           = tf.keras.layers.Dropout(0.2,  name='dropout_expl_1')(x)
    x           = tf.keras.layers.Dense(64,  activation='relu', name='dense_expl_2')(x)
    x           = tf.keras.layers.Dropout(0.2,  name='dropout_expl_2')(x)
    expl_output = tf.keras.layers.Dense(1,   activation='sigmoid', name='output_expl')(x)
    dl_explainer_model = tf.keras.Model(inputs=expl_input, outputs=expl_output)

    # FIX 3: dl_model is already defined above, so these get_layer calls are safe.
    try:
        dl_explainer_model.get_layer('dense_expl_1').set_weights(
            dl_model.get_layer('dense_3').get_weights()
        )
        dl_explainer_model.get_layer('dense_expl_2').set_weights(
            dl_model.get_layer('dense_4').get_weights()
        )
        dl_explainer_model.get_layer('output_expl').set_weights(
            dl_model.get_layer('dense_5').get_weights()
        )
    except (ValueError, KeyError) as e:
        st.warning(
            f"Could not transfer weights to explainer model — SHAP/LIME results "
            f"may be inaccurate. Check dense layer names in your saved model. Error: {e}"
        )

    feature_names_shap = (
        [f'user_emb_{i}' for i in range(embedding_dim)] +
        [f'item_emb_{i}' for i in range(embedding_dim)]
    )

    # ---- SHAP background dataset ----
    if 'user_idx' not in synthetic_ratings_df.columns:
        synthetic_ratings_df['user_idx'] = synthetic_ratings_df['user_id'].map(user_id_to_idx)
    if 'item_idx' not in synthetic_ratings_df.columns:
        synthetic_ratings_df['item_idx'] = synthetic_ratings_df['item_id']

    num_bg = min(100, len(synthetic_ratings_df))
    bg_idx  = np.random.choice(len(synthetic_ratings_df), num_bg, replace=False)

    bg_user_idx  = synthetic_ratings_df.iloc[bg_idx]['user_idx'].values
    bg_item_idx  = synthetic_ratings_df.iloc[bg_idx]['item_idx'].values

    bg_user_emb  = dl_model.get_layer('user_embedding').get_weights()[0][bg_user_idx]
    bg_item_emb  = dl_model.get_layer('item_embedding').get_weights()[0][bg_item_idx]
    X_background_shap = np.concatenate([bg_user_emb, bg_item_emb], axis=1)

    # Return everything in a single dict — no positional unpacking, no scope issues.
    return dict(
        df=df,
        df_fe=df_fe,
        cosine_sim=cosine_sim,
        synthetic_ratings_df=synthetic_ratings_df,
        dl_model=dl_model,
        faiss_index=faiss_index,
        eval_results=eval_results,
        st_model=st_model,
        song_to_item_id=song_to_item_id,
        item_id_to_song=item_id_to_song,
        user_id_to_idx=user_id_to_idx,
        idx_to_user_id=idx_to_user_id,
        num_users=num_users,
        num_items=num_items,
        algo_svd=algo_svd,
        dl_explainer_model=dl_explainer_model,
        feature_names_shap=feature_names_shap,
        X_background_shap=X_background_shap,
        embedding_dim=embedding_dim,
    )


# ---- Unpack artifact dict into module-level names ----
_A = load_all_artifacts()

df                   = _A['df']
df_fe                = _A['df_fe']
cosine_sim           = _A['cosine_sim']
synthetic_ratings_df = _A['synthetic_ratings_df']
dl_model             = _A['dl_model']
faiss_index          = _A['faiss_index']
eval_results         = _A['eval_results']
st_model             = _A['st_model']
song_to_item_id      = _A['song_to_item_id']
item_id_to_song      = _A['item_id_to_song']
user_id_to_idx       = _A['user_id_to_idx']
idx_to_user_id       = _A['idx_to_user_id']
num_users            = _A['num_users']
num_items            = _A['num_items']
algo_svd             = _A['algo_svd']
dl_explainer_model   = _A['dl_explainer_model']
feature_names_shap   = _A['feature_names_shap']
X_background_shap    = _A['X_background_shap']
embedding_dim        = _A['embedding_dim']


# =============================================================================
# Recommendation helpers
# =============================================================================

def get_content_based_recommendations(song_title, num_recommendations=5):
    try:
        idx = df[df['Song-Name'] == song_title].index[0]
    except IndexError:
        return f"Song '{song_title}' not found in the dataset."
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    return df['Song-Name'].iloc[[i for i, _ in sim_scores]].tolist()


def get_cf_recommendations(user_id, num_recommendations=5):
    all_item_ids = list(item_id_to_song.keys())
    if (user_id not in user_id_to_idx or
            user_id not in algo_svd.trainset._inner2raw_id_users):
        st.warning(f"User '{user_id}' not in CF training data. Returning popular songs.")
        return df['Song-Name'].value_counts().head(num_recommendations).index.tolist()

    inner_uid = algo_svd.trainset.to_inner_uid(user_id)
    rated_raw = {algo_svd.trainset.to_raw_iid(ii) for ii, _ in algo_svd.trainset.ur[inner_uid]}
    preds = sorted(
        [(iid, algo_svd.predict(user_id, iid).est)
         for iid in all_item_ids if iid not in rated_raw],
        key=lambda x: x[1], reverse=True
    )
    return [item_id_to_song[iid] for iid, _ in preds[:num_recommendations]]


def get_hybrid_recommendations(user_id, song_title, num_recommendations=5,
                                cf_weight=0.6, content_weight=0.4):
    if round(cf_weight + content_weight, 2) != 1.0:
        st.error("CF and Content-Based weights must sum to 1.0")
        return []

    all_item_ids = list(item_id_to_song.keys())
    cf_map, rated_raw = {}, set()

    if (user_id in user_id_to_idx and
            user_id in algo_svd.trainset._inner2raw_id_users):
        inner_uid = algo_svd.trainset.to_inner_uid(user_id)
        rated_raw = {algo_svd.trainset.to_raw_iid(ii)
                     for ii, _ in algo_svd.trainset.ur[inner_uid]}
        for iid in all_item_ids:
            if iid not in rated_raw:
                cf_map[item_id_to_song[iid]] = algo_svd.predict(user_id, iid).est
    else:
        for iid in all_item_ids:
            cf_map[item_id_to_song[iid]] = 5.0

    try:
        idx = df[df['Song-Name'] == song_title].index[0]
    except IndexError:
        st.warning(f"Song '{song_title}' not found for content-based blending.")
        return []

    content_map = {df['Song-Name'].iloc[i]: s for i, s in enumerate(cosine_sim[idx])}

    scores = {}
    for name in df['Song-Name'].unique():
        if name == song_title:
            continue
        norm_cf = (cf_map.get(name, 5.0) - 1) / 9.0
        scores[name] = cf_weight * norm_cf + content_weight * content_map.get(name, 0.0)

    final = []
    for name, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if name in song_to_item_id and song_to_item_id[name] in rated_raw:
            continue
        final.append(name)
        if len(final) >= num_recommendations:
            break
    return final


def get_dl_recommendations(user_id, num_recommendations=5):
    if user_id not in user_id_to_idx:
        st.warning(f"User '{user_id}' not found for Deep Learning recommendations.")
        return []

    user_idx   = user_id_to_idx[user_id]
    item_idxs  = np.arange(num_items)
    user_batch = np.full(num_items, user_idx)

    pred_norm = dl_model.predict([user_batch, item_idxs], verbose=0)
    min_r, max_r = synthetic_ratings_df['rating'].min(), synthetic_ratings_df['rating'].max()
    pred = pred_norm * (max_r - min_r) + min_r

    if 'user_idx' not in synthetic_ratings_df.columns:
        synthetic_ratings_df['user_idx'] = synthetic_ratings_df['user_id'].map(user_id_to_idx)
    if 'item_idx' not in synthetic_ratings_df.columns:
        synthetic_ratings_df['item_idx'] = synthetic_ratings_df['item_id']

    rated = set(synthetic_ratings_df[
        synthetic_ratings_df['user_id'] == user_id]['item_idx'].tolist())

    recs = []
    for iidx, score in sorted(zip(item_idxs, pred.flatten()),
                               key=lambda x: x[1], reverse=True):
        if iidx not in rated:
            recs.append((item_id_to_song[iidx], score))
            if len(recs) >= num_recommendations:
                break
    return recs


def faiss_search_recommendations(query_text, num_recommendations=5, top_n_faiss=50):
    if not query_text:
        return []
    q_emb = st_model.encode([query_text]).astype('float32')
    _, indices = faiss_index.search(q_emb, top_n_faiss)
    out = []
    for i in indices[0]:
        if i < len(df_fe) and df_fe['Song-Name'].iloc[i].lower() != query_text.lower():
            out.append(i)
        if len(out) >= num_recommendations:
            break
    return df_fe['Song-Name'].iloc[out].tolist()


# =============================================================================
# Explanation helpers
# =============================================================================

def get_content_based_explanations(query_song_title, recommended_songs):
    explanations = []
    try:
        q_idx = df[df['Song-Name'] == query_song_title].index[0]
    except IndexError:
        return [f"Query song '{query_song_title}' not found for explanation."]

    genre_cols = [c for c in ['Dance', 'Bollywood', 'Romantic', 'Devotional', 'Sad',
                               'Motivational', 'Romance', 'Sensual', 'Patriotic']
                  if c in df_fe.columns]
    q_genres = [c for c in genre_cols if df_fe.loc[q_idx, c] == 1]

    for rec in recommended_songs:
        try:
            r_idx = df[df['Song-Name'] == rec].index[0]
        except IndexError:
            explanations.append(f"Song '{rec}' not found for explanation.")
            continue
        common = list(set(q_genres) & {c for c in genre_cols if df_fe.loc[r_idx, c] == 1})
        text = f"**'{rec}'** is recommended because:\n"
        text += (f"  ✓ **Similar Genre(s):** {', '.join(common)}\n" if common
                 else f"  ✓ **Similar Genre Profile** with '{query_song_title}'.\n")
        text += "  ✓ **Semantic Similarity** via NLP embeddings.\n"
        explanations.append(text)
    return explanations


def get_cf_explanations(user_id, recommended_songs):
    top_rated = []
    if user_id in user_id_to_idx:
        ur = synthetic_ratings_df[synthetic_ratings_df['user_id'] == user_id].copy()
        ur['song_name'] = ur['item_id'].map(item_id_to_song)
        if not ur.empty:
            top_rated = ur.sort_values('rating', ascending=False).head(3)['song_name'].tolist()

    liked = ', '.join(top_rated) if top_rated else 'many songs'
    exps  = [f"For user **'{user_id}'** who enjoys: {liked}:\n"]
    for song in recommended_songs:
        exps.append(
            f"**'{song}'** is recommended because:\n"
            "  ✓ Users with similar taste patterns have rated it highly.\n"
        )
    return exps


# FIX 4: dl_shap_predict references dl_explainer_model which is now a proper
#         module-level variable, guaranteed to be set before any button click.
def dl_shap_predict(X):
    return dl_explainer_model.predict(X, verbose=0)


@st.cache_data(show_spinner=False)
def get_dl_explanations_shap(user_id, song_name, num_features=10):
    if user_id not in user_id_to_idx or song_name not in song_to_item_id:
        return ["Invalid user or song for SHAP explanation."]
    if X_background_shap.shape[0] < 2:
        return ["Insufficient background data for SHAP."]

    u_emb = dl_model.get_layer('user_embedding').get_weights()[0][user_id_to_idx[user_id]]
    i_emb = dl_model.get_layer('item_embedding').get_weights()[0][song_to_item_id[song_name]]
    X_exp = np.concatenate([u_emb, i_emb]).reshape(1, -1)

    explainer   = shap.KernelExplainer(dl_shap_predict, X_background_shap)
    shap_values = explainer.shap_values(X_exp)
    sv = (shap_values[0][0] if isinstance(shap_values, list) else shap_values[0]).flatten()

    top_idx = np.argsort(np.abs(sv))[::-1]
    lines   = [f"**SHAP — '{song_name}' for '{user_id}':**",
               "Top contributing embedding dimensions:"]
    for i in range(min(num_features, len(top_idx))):
        lines.append(f"  ✓ {feature_names_shap[top_idx[i]]}: {sv[top_idx[i]]:.4f}")
    lines.append(
        "**Interpretation:** Positive values push the predicted rating up; "
        "negative values push it down."
    )
    return lines


@st.cache_data(show_spinner=False)
def get_dl_explanations_lime(user_id, song_name, num_features=10):
    if user_id not in user_id_to_idx or song_name not in song_to_item_id:
        return ["Invalid user or song for LIME explanation."]
    if X_background_shap.shape[0] < 2:
        return ["Insufficient background data for LIME."]

    u_emb = dl_model.get_layer('user_embedding').get_weights()[0][user_id_to_idx[user_id]]
    i_emb = dl_model.get_layer('item_embedding').get_weights()[0][song_to_item_id[song_name]]
    X_exp = np.concatenate([u_emb, i_emb])

    lime_exp = lime_tabular.LimeTabularExplainer(
        training_data=X_background_shap,
        feature_names=feature_names_shap,
        class_names=['predicted_rating'],
        mode='regression'
    ).explain_instance(X_exp, dl_shap_predict, num_features=num_features)

    lines = [f"**LIME — '{song_name}' for '{user_id}':**",
             "Top contributing embedding dimensions:"]
    for feat, w in lime_exp.as_list():
        lines.append(f"  ✓ {feat}: {w:.4f}")
    lines.append(
        "**Interpretation:** Positive weights increase the predicted rating; "
        "negative weights decrease it."
    )
    return lines



# =============================================================================
# Custom CSS — Dark Neon Theme
# =============================================================================
st.markdown("""
<style>
/* ── Global Styles ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #090A1A !important; /* Deep Midnight Blue */
    color: #F4F4F6 !important;            /* Soft White */
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #090A1A !important;
    border-right: 1px solid #FFD70033; /* Gold with transparency */
}
[data-testid="stSidebar"] * { color: #F4F4F6 !important; }

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 { color: #FFD700 !important; } /* Warm Gold */

/* ── Buttons ── */
.stButton > button {
    background-color: transparent !important;
    border: 2px solid #FFD700 !important;
    color: #FFD700 !important;
    font-weight: 600;
    border-radius: 8px;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #FFD700 !important;
    color: #090A1A !important;
}

/* ── Dropdowns / Inputs ── */
[data-testid="stSelectbox"] > div > div, 
[data-testid="stTextInput"] input {
    background-color: #1a1c32 !important;
    border: 1px solid #FFD70055 !important;
    color: #F4F4F6 !important;
    border-radius: 8px;
}

/* ── Song Cards ── */
.song-card {
    background: #161829;
    border: 1px solid #FFD70033;
    border-radius: 12px;
    padding: 14px;
    margin: 8px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.song-number { color: #FFD700; font-weight: bold; }
.song-name { color: #F4F4F6; }
.song-score { 
    margin-left: auto; 
    color: #FFD700; 
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


def song_card(rank, name, score=None):
    score_html = f'<span class="song-score">★ {score:.2f}</span>' if score is not None else ""
    return f'''
    <div class="song-card">
        <span class="song-number">#{rank}</span>
        <span class="song-name">🎵 {name}</span>
        {score_html}
    </div>
    '''


def expl_box(text):
    return f'<div class="expl-box">{text}</div>'


st.markdown("# 🎵 Instant Music Recommender")
st.markdown("<p style='text-align: center; color: #E0E2E7;'>Select a song to discover similar tracks.</p>", unsafe_allow_html=True)

# Create a layout that centers the content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    selected_song = st.selectbox("🎼 Select a song:", df['Song-Name'].unique().tolist())
    
    # Simple, direct action
    if st.button("🚀 Recommend Similar Songs"):
        with st.spinner("Finding similar songs…"):
            recs = get_content_based_recommendations(selected_song, 5)
            
            if isinstance(recs, list) and recs:
                st.markdown(f"### Songs similar to *{selected_song}*")
                for i, s in enumerate(recs, 1):
                    # Using your existing song_card function
                    st.markdown(song_card(i, s), unsafe_allow_html=True)
            else:
                st.error("No recommendations found.")

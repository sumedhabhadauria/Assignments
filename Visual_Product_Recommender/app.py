import pickle
from pathlib import Path
import time

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(
    page_title="Visual Product Recommender",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TOP_K = 20

BASE_DIR    = Path(__file__).resolve().parent
MODELS_DIR  = BASE_DIR / "Models"
OUTPUTS_DIR = BASE_DIR / "Outputs"

FEATURE_EXTRACTOR_PATH = MODELS_DIR / "feature_extractor.keras"
FAISS_INDEX_PATH       = MODELS_DIR / "faiss_index.index"
EMBEDDINGS_PATH        = OUTPUTS_DIR / "embeddings.npy"
IMAGE_IDS_PATH         = OUTPUTS_DIR / "image_ids.pkl"
CSV_PATH               = OUTPUTS_DIR / "clean_metadata.csv"
DATASET_FOLDER         = BASE_DIR / "images"

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #F5F7FA;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
#MainMenu, footer, header        { visibility: hidden; }
[data-testid="stDecoration"]     { display: none; }
[data-testid="stSidebar"]        { display: none; }
[data-testid="collapsedControl"] { display: none; }

.topbar {
    background: #1A2B4A;
    padding: 18px 36px;
    border-radius: 0 0 10px 10px;
    margin-bottom: 28px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.18);
}
.topbar-brand { font-size: 1.3rem; font-weight: 700; color: #fff; }

/* metric row */
.metric-row { display:flex; gap:14px; margin:18px 0 24px; flex-wrap:wrap; }
.mcard {
    background:#fff; border:1px solid #DDE3ED; border-radius:10px;
    padding:14px 20px; flex:1; min-width:130px;
    box-shadow:0 1px 4px rgba(0,0,0,0.05);
}
.mcard .ml { font-size:.70rem; font-weight:700; color:#6B7A99;
             text-transform:uppercase; letter-spacing:.8px; margin-bottom:4px; }
.mcard .mv { font-size:1.35rem; font-weight:700; color:#1A2B4A; }
.mcard .ms { font-size:.72rem; color:#2E7DF7; margin-top:2px; }

/* section header */
.sec { display:flex; align-items:center; gap:10px; margin:20px 0 14px; }
.sec h2 { font-size:.85rem; font-weight:700; color:#1A2B4A;
          margin:0; text-transform:uppercase; letter-spacing:1px; }
.secline { flex:1; height:1px; background:#DDE3ED; }

/* result card meta */
.rcard-body { padding:10px 12px; background:#fff;
              border:1px solid #DDE3ED; border-top:none;
              border-radius:0 0 10px 10px; }
.rcard-rank { font-size:.68rem; font-weight:700; color:#2E7DF7;
              text-transform:uppercase; letter-spacing:.6px; }
.rcard-name { font-size:.85rem; font-weight:700; color:#1A2B4A;
              margin:3px 0 2px; overflow:hidden;
              display:-webkit-box; -webkit-line-clamp:2;
              -webkit-box-orient:vertical; }
.rcard-meta { font-size:.73rem; color:#6B7A99; margin-bottom:6px; }
.sim-badge  { display:inline-block; padding:2px 9px; border-radius:20px;
              font-size:.72rem; font-weight:700;
              background:#E8F0FE; color:#1967D2; }
.bar-bg   { background:#EEF2F8; border-radius:4px; height:5px;
            overflow:hidden; margin-top:6px; }
.bar-fill { background:linear-gradient(90deg,#2E7DF7,#5BA8FF);
            height:5px; border-radius:4px; }

/* query box */
.query-wrap { background:#fff; border:1px solid #DDE3ED;
              border-radius:10px; padding:14px;
              box-shadow:0 1px 4px rgba(0,0,0,0.05); }
.query-label { font-size:.70rem; font-weight:700; color:#6B7A99;
               text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }

/* button */
.stButton > button {
    background:#2E7DF7 !important; color:#fff !important;
    border:none !important; border-radius:8px !important;
    font-weight:600 !important; padding:10px 32px !important;
    font-size:.9rem !important; width:100%;
}
.stButton > button:hover { background:#1A6AE0 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_resources():
    feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
    embeddings        = np.load(EMBEDDINGS_PATH).astype("float32")
    with open(IMAGE_IDS_PATH, "rb") as f:
        image_ids = pickle.load(f)
    metadata = pd.read_csv(CSV_PATH)
    index    = faiss.read_index(str(FAISS_INDEX_PATH))
    return feature_extractor, embeddings, image_ids, metadata, index

with st.spinner():
    feature_extractor, embeddings, image_ids, metadata, index = load_resources()

# FEATURE EXTRACTION
def extract_feature(uploaded_image) -> np.ndarray:
    img = Image.open(uploaded_image).convert("RGB")
    img = img.resize((224, 224))
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature = feature_extractor.predict(img, verbose=0).flatten()
    feature = feature / np.linalg.norm(feature)
    return feature.astype("float32")

# SEARCH 
def search_products(feature_vector: np.ndarray, top_k: int = TOP_K):
    distances, indices = index.search(
        np.expand_dims(feature_vector, axis=0), top_k
    )
    recommendations = []
    for idx, dist in zip(indices[0], distances[0]):
        product_id = int(image_ids[idx])
        rows = metadata[metadata["id"] == product_id]
        if rows.empty:
            continue
        row = rows.iloc[0]
        cosine_sim = float(np.clip(1.0 - (dist ** 2) / 2.0, 0.0, 1.0))
        recommendations.append({
            "id":         product_id,
            "distance":   float(dist),
            "similarity": cosine_sim,
            "name":       str(row.get("productDisplayName", product_id)),
            "gender":     str(row.get("gender", "—")),
            "category":   str(row.get("masterCategory", "—")),
            "subcategory":str(row.get("subCategory", "—")),
            "article":    str(row.get("articleType", "—")),
            "colour":     str(row.get("baseColour", "—")),
        })
    return recommendations

st.markdown("""
<div class="topbar">
  <div class="topbar-brand">Visual Product Recommender</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if not uploaded_file:
    st.markdown("""
    <div style="text-align:center;padding:72px 20px;color:#6B7A99;">
      <div style="font-size:3rem;margin-bottom:12px;">🛍️</div>
      <div style="font-size:1.05rem;font-weight:600;color:#1A2B4A;margin-bottom:6px;">
        Upload a fashion product image to find similar items
      </div>
      <div style="font-size:.82rem;">
        Supports apparel, footwear, accessories and more.
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

col_q, col_b = st.columns([1, 3], gap="large")
with col_q:
    st.markdown('<div class="query-wrap"><div class="query-label">Your Product</div>',
                unsafe_allow_html=True)
    st.image(uploaded_file, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col_b:
    st.write("")
    run = st.button("Find Similar Products")

if not run:
    st.stop()

t0 = time.time()
with st.spinner("Extracting visual features…"):
    feature_vector = extract_feature(uploaded_file)
embed_ms = round((time.time() - t0) * 1000, 1)

t1 = time.time()
with st.spinner(f"Finding top {TOP_K} similar products…"):
    recommendations = search_products(feature_vector, top_k=TOP_K)
search_ms = round((time.time() - t1) * 1000, 1)

if not recommendations:
    st.warning("No results found. Try a different image.")
    st.stop()


st.markdown("""
<div class="sec">
  <span style="font-size:1rem"></span>
  <h2>Similar Products</h2>
  <div class="secline"></div>
</div>""", unsafe_allow_html=True)

COLS = 5
for row_start in range(0, len(recommendations), COLS):
    row  = recommendations[row_start: row_start + COLS]
    cols = st.columns(COLS, gap="medium")
    for col, prod in zip(cols, row):
        with col:
            pct        = int(prod["similarity"] * 100)
            image_path = DATASET_FOLDER / f"{prod['id']}.jpg"

            if image_path.exists():
                st.image(Image.open(image_path), use_container_width=True)
            else:
                st.markdown(f"""
                <div style="background:#EEF2F8;height:160px;border-radius:10px 10px 0 0;
                     display:flex;align-items:center;justify-content:center;
                     color:#6B7A99;font-size:.78rem;">
                  ID: {prod['id']}
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="rcard-body">
              <div class="rcard-rank">Rank #{row_start + recommendations.index(prod) + 1}</div>
              <div class="rcard-name">{prod['name']}</div>
              <div class="rcard-meta">
                {prod['article']} &nbsp;·&nbsp; {prod['gender']} &nbsp;·&nbsp; {prod['colour']}
              </div>
              <span class="sim-badge">{pct}% match</span>
              <div class="bar-bg">
                <div class="bar-fill" style="width:{pct}%;"></div>
              </div>
            </div>""", unsafe_allow_html=True)
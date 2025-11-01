import streamlit as st
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# --- Page setup ---
st.set_page_config(page_title="Text Similarity App", layout="wide")
st.title("🧠 Text Similarity Explorer")
st.markdown(
    "Compare two texts using **SentenceTransformer embeddings** and **BERTScore (F1)**."
)

# --- Load SentenceTransformer model (cached) ---
@st.cache_resource(show_spinner=True)
def load_model():
    st.info("Loading model 'sentence-transformers/all-MiniLM-L6-v2' ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    st.success("✅ Model loaded successfully!")
    return model

model = load_model()

# --- Text inputs ---
col1, col2 = st.columns(2)
with col1:
    text1 = st.text_area("Text 1 (Ground Truth)", "The apple is red.")
with col2:
    text2 = st.text_area("Text 2 (Compared Text)", "The fruit is red and round.")

# --- Compute similarity and BERTScore ---
if st.button("Compute similarity"):
    if not text1 or not text2:
        st.warning("Please enter both texts.")
    else:
        with st.spinner("Computing embeddings and BERTScore..."):
            # Compute embeddings
            emb1 = model.encode(text1, convert_to_tensor=True)
            emb2 = model.encode(text2, convert_to_tensor=True)

            # Cosine similarity
            sim = cosine_similarity(
                emb1.cpu().numpy().reshape(1, -1),
                emb2.cpu().numpy().reshape(1, -1)
            )[0][0]

            # BERTScore (English)
            P, R, F1 = bert_score(
                [text2], [text1], lang="en", verbose=False, rescale_with_baseline=True
            )

        # --- Display results ---
        st.subheader("Results")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Cosine Similarity", f"{sim:.4f}")
        with c2:
            st.metric("BERTScore (F1)", f"{F1.mean().item():.4f}")

        st.markdown("---")
        st.markdown(f"**Precision (P):** {P.mean().item():.4f}  |  **Recall (R):** {R.mean().item():.4f}")

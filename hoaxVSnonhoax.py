import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import base64

# ======================================================
# 1. Page Config
# ======================================================
st.set_page_config(
    page_title="Hoax Detector IndoBERT",
    page_icon="🛰️",
    layout="centered"
)

st.markdown("""
<style>
body { 
    background: linear-gradient(180deg, #e2f0ff, #ffffff);
}
.title {
    font-size: 40px; 
    text-align: center;
    font-weight: bold;
}
.sub {
    text-align: center;
    font-size: 18px;
}
.box {
    padding: 15px;
    border-radius: 15px;
    background-color: #f5f9ff;
    border: 1px solid #d4e4ff;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 2. Load Model from HuggingFace
# ======================================================
HF_MODEL = "mcwazawski/bjirrrrcoy"  # ← Ganti dengan model ID tepat dari HF

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(HF_MODEL)
    model = BertForSequenceClassification.from_pretrained(HF_MODEL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device

tokenizer, model, device = load_model()

# ======================================================
# 3. Cleaning Function
# ======================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ======================================================
# 4. Predict Function
# ======================================================
def predict(text):
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()

    return pred, cleaned

# ======================================================
# 5. Animations (GIF encoded)
# ======================================================
def load_gif(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

gif_hoax = load_gif("./hoax.gif")
gif_nonhoax = load_gif("./nonhoax.gif")

# ======================================================
# 6. UI
# ======================================================
st.markdown("<p class='title'>🛰️ HOAX DETECTOR with IndoBERT</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Deteksi hoax secara otomatis dengan model BERT yang sudah kamu latih</p>", unsafe_allow_html=True)

caption = st.text_area(" ", height=150, placeholder="Contoh: Vaksin membuat badan jadi magnet...")

btn = st.button("🔍 Deteksi Sekarang")

# ======================================================
# 7. Prediction Output
# ======================================================
if btn:
    if caption.strip() == "":
        st.warning("⚠️ Teks masih kosong!")
    else:
        label, cleaned = predict(caption)

        st.write("### 🔧 Teks setelah dibersihkan:")
        st.info(cleaned)

        st.write("---")

        if label == 1:
            st.markdown("## 🔥 Hasil: **HOAX TERDETEKSI!** 🚨")
            st.markdown(
                f"<img src='data:image/gif;base64,{gif_hoax}' width='350'/>",
                unsafe_allow_html=True
            )
            st.error("⚠️ Caption ini terindikasi **Hoax** berdasarkan model IndoBERT")
        else:
            st.markdown("## 🟢 Hasil: **FAKTA / NON-HOAX** 🎉")
            st.markdown(
                f"<img src='data:image/gif;base64,{gif_nonhoax}' width='350'/>",
                unsafe_allow_html=True
            )
            st.success("Caption ini terindikasi **Tidak Mengandung Hoax**")

        st.write("---")
        st.caption("Model: IndoBERT Sequence Classification — Fine-tuned for Hoax Detection")

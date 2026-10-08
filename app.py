import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="AI 垃圾分類助手", page_icon="♻️")
st.title("♻️ 本地端 AI 垃圾分類系統")

# 1. 載入本地訓練好的分類模型
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
    st.success("✅ 專屬 AI 分類模型已成功載入")
except Exception as e:
    st.error(f"⚠️ 找不到 best.pt，請確認檔案已放在主目錄：{e}")

# 2. 圖片上傳與辨識
uploaded_file = st.file_uploader("請上傳垃圾照片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="上傳的照片", use_container_width=True)
    
    if st.button("🔍 開始 AI 辨識"):
        with st.spinner("模型辨識中..."):
            results = model(image)
            
            # 取得分類結果與信心度
            top1_id = results[0].probs.top1
            class_name = results[0].names[top1_id]
            confidence = float(results[0].probs.top1conf) * 100
            
            st.metric(label="辨識結果類別", value=class_name)
            st.write(f"**辨識信心度**：`{confidence:.2f}%`")
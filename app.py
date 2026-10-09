import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 頁面基本設定
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
    st.error(f"⚠️ 載入模型失敗：{e}")

# 2. 圖片上傳與辨識
uploaded_file = st.file_uploader("請上傳垃圾照片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="上傳的照片", use_container_width=True)
    
    if st.button("🔍 開始 AI 辨識"):
        with st.spinner("模型辨識中..."):
            results = model(image)
            
            # 取得模型辨識結果
            top1_id = results[0].probs.top1
            class_name = results[0].names[top1_id]
            confidence = float(results[0].probs.top1conf) * 100
            
            # 成功提示與排版呈現
            st.success("分析完成！")
            st.markdown("這張圖片中的主要物品與相關分類說明如下：")
            
            st.markdown(f"1. **物品名稱**：`{class_name}` （辨識信心度：`{confidence:.2f}%`）")
            
            st.markdown("2. **垃圾分類類別**：")
            st.markdown(f"   - **模型判定結果**：`{class_name}`")
            st.markdown("   - **乾淨且乾燥**：對應資源回收類處理")
            st.markdown("   - **沾有油污／液體**：建議改列為一般垃圾處理")
            
            st.markdown("3. **處理建議**：")
            st.markdown("   - **重複利用（最推薦）**：若容器或物品完整乾淨，建議留著作為二次使用，延長其壽命。")
            st.markdown("   - **回收處理（若乾淨）**：必須確保內部無雜物、無油漬且保持乾燥後，交給資源回收車。")
            st.markdown("   - **丟棄處理（若髒污）**：若曾裝過食物、沾有油污或無法徹底沖洗乾淨，請直接當作「一般垃圾」丟棄。")
            
            # 底部保留原本的排查除錯區塊（方便專題評審或除錯時檢視）
            with st.expander("📊 完整機率分布與除錯資訊"):
                st.write("模型類別名稱列表：", results[0].names)
                st.write("各類別預測得分：", results[0].probs.data.tolist())
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 頁面基本設定
st.set_page_config(page_title="本地端 AI 垃圾分類系統", page_icon="♻️")
st.title("♻️ 本地端 AI 垃圾分類系統")

# 1. 載入本地訓練好的 YOLO 分類模型
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
    st.success("✅ 專屬 AI 分類模型已成功載入")
except Exception as e:
    st.error(f"⚠️ 載入模型失敗：{e}")

# 2. 定義針對不同物品的詳細說明字典（可根據你們 YOLO 訓練的類別名稱自行增減修改）
advice_database = {
    "塑膠袋": {
        "name": "塑膠袋（粉白條紋背心袋／購物袋）",
        "category": "- **乾淨且乾燥**：資源回收 - 塑膠袋類\n   - **沾有油污／液體／血水**：一般垃圾",
        "tips": "- **重複利用（最推薦）**：建議留著作為二次購物袋或家中的垃圾袋使用，延長其壽命。\n   - **回收處理（若乾淨）**：必須確保袋內無雜物、無油漬且保持乾燥，可將其鋪平或打結打包後，交給資源回收車。\n   - **丟棄處理（若髒污）**：若塑膠袋曾裝過食物、沾有油污或無法徹底沖洗乾淨，請直接當作「一般垃圾」丟棄，以免污染其他回收物。"
    },
    "寶特瓶": {
        "name": "寶特瓶（PET 塑膠容器）",
        "category": "- **乾淨並去除瓶蓋與標籤**：資源回收 - 塑膠容器類\n   - **沾染化學液體或無法清洗**：一般垃圾",
        "tips": "- **回收前準備**：請先將瓶內殘留液體倒乾、簡單沖洗，並將塑膠膜/標籤撕除。\n   - **體積壓縮**：建議將寶特瓶壓扁，以減少回收桶的佔用空間。\n   - **交付回收**：投入指定塑膠類資源回收桶或交由清潔隊回收車。"
    },
    "紙箱": {
        "name": "紙箱／紙類包裝",
        "category": "- **乾淨無油漬**：資源回收 - 紙容器／紙類\n   - **沾有油漬（如披薩盒）**：一般垃圾",
        "tips": "- **拆解壓平**：請將紙箱膠帶撕除並將其摺疊壓平，以利回收與運送。\n   - **保持乾燥**：紙類受潮或沾染油污後無法回收，若有油漬請作為一般垃圾處理。"
    }
}

# 預設的通用說明（當 YOLO 辨識出字典裡沒有的其他類別時使用）
default_advice = {
    "category": "- 請根據當地環保局規範進行分類",
    "tips": "- **重複利用**：若物品完整乾淨，建議延長其使用壽命。\n   - **妥善處理**：請依材質（如紙類、塑膠、金屬）投入對應的回收桶，若無法判定或嚴重髒污則作為一般垃圾處理。"
}

# 3. 圖片上傳與辨識
uploaded_file = st.file_uploader("請上傳垃圾照片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="上傳的照片", use_container_width=True)
    
    if st.button("🔍 開始 AI 辨識"):
        with st.spinner("模型辨識中..."):
            results = model(image)
            
            # 取得最高信心度的類別與數值
            top1_id = results[0].probs.top1
            class_name = results[0].names[top1_id]
            confidence = float(results[0].probs.top1conf) * 100
            
            # 取得對應的詳細說明（若找不到對應名稱則使用預設值）
            item_info = advice_database.get(class_name, {
                "name": class_name,
                "category": f"- **模型判定類別**：{class_name}\n   - 請依實際材質判定為資源回收或一般垃圾",
                "tips": default_advice["tips"]
            })
            
            # 呈現符合你要求的專刊排版
            st.success("分析完成！")
            st.markdown("這張圖片中的主要物品與相關分類說明如下：")
            
            st.markdown(f"1. **物品名稱**：{item_info['name']} （辨識信心度：`{confidence:.2f}%`）")
            
            st.markdown("2. **垃圾分類類別**：")
            st.markdown(item_info['category'])
            
            st.markdown("3. **處理建議**：")
            st.markdown(item_info['tips'])
            
            # 排查除錯用：列出所有類別對應與機率
            with st.expander("📊 完整機率分布與除錯資訊"):
                st.write("模型類別名稱列表：", results[0].names)
                st.write("各類別預測得分：", results[0].probs.data.tolist())
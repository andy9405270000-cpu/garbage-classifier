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

# 2. 建立一個智慧動態說明生成函數（根據 YOLO 辨識出的類別名稱自動延伸）
def generate_dynamic_advice(class_name):
    lower_name = class_name.lower()
    
    # 針對金屬類 (Metal) 的動態排版
    if "metal" in lower_name or "金屬" in class_name or "鐵" in class_name or "鋁" in class_name:
        return {
            "name": f"金屬類物品 ({class_name})",
            "category": "- **乾淨無嚴重鏽蝕**：資源回收 - 金屬類／鐵鋁罐\n   - **沾有有害化學物質或醫療針具**：特殊有害垃圾",
            "tips": "- **清洗與壓扁**：建議先將內容物倒乾、簡單沖洗，若是鐵鋁罐可稍微踩扁以減少體積。\n   - **回收處理**：金屬材質可無限次循環利用，請直接投入金屬資源回收桶。\n   - **安全注意事項**：若有尖銳邊緣，請妥善包覆後再丟棄，避免割傷清潔人員。"
        }
    # 針對塑膠類 (Plastic) 的動態排版
    elif "plastic" in lower_name or "塑膠" in class_name or "bottle" in lower_name or "瓶" in class_name:
        return {
            "name": f"塑膠類容器／物品 ({class_name})",
            "category": "- **乾淨且乾燥**：資源回收 - 塑膠類\n   - **嚴重油污或複合材質**：一般垃圾",
            "tips": "- **清洗準備**：請先將內部殘留物清除並沖洗乾淨。\n   - **去除配件**：建議將瓶蓋與塑膠標籤撕除（部分回收廠分開處理效率更好）。\n   - **回收交付**：投入塑膠資源回收桶或交由清潔隊回收車。"
        }
    # 針對紙類 (Paper) 的動態排版
    elif "paper" in lower_name or "紙" in class_name or "box" in lower_name or "箱" in class_name:
        return {
            "name": f"紙類／紙箱包裝 ({class_name})",
            "category": "- **乾淨無油漬**：資源回收 - 紙類／紙容器\n   - **沾染油漬或受潮嚴重**：一般垃圾",
            "tips": "- **拆解壓平**：請將紙箱膠帶撕除，並將其摺疊壓平以節省空間。\n   - **保持乾燥**：紙類若沾有油污（如披薩盒的油漬）將無法回收，請改投入一般垃圾。"
        }
    # 針對其他所有未定義類別的通用智慧排版
    else:
        return {
            "name": f"{class_name}",
            "category": f"- **材質判定**：經模型辨識為 `{class_name}`\n   - **一般判定標準**：乾淨可回收，髒污或複合材質請改列一般垃圾",
            "tips": f"- **重複利用**：若該 `{class_name}` 物品仍完整乾淨，建議優先考慮二次使用或延長其壽命。\n   - **妥善處理**：請依當地環保局分類規範，將其投入對應的資源回收桶；若無法判定材質或受損嚴重，則作為一般垃圾處理。"
        }

# 3. 圖片上傳與辨識
uploaded_file = st.file_uploader("請上傳垃圾照片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="上傳的照片", use_container_width=True)
    
    if st.button("🔍 開始 AI 辨識"):
        with st.spinner("模型辨識與動態分析中..."):
            results = model(image)
            
            # 取得最高信心度的類別與數值
            top1_id = results[0].probs.top1
            class_name = results[0].names[top1_id]
            confidence = float(results[0].probs.top1conf) * 100
            
            # 取得動態生成的詳細說明
            item_info = generate_dynamic_advice(class_name)
            
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
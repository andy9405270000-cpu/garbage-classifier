if st.button("🔍 開始 AI 辨識"):
    with st.spinner("模型辨識中..."):
        # 執行辨識
        results = model(image)
        
        # 正確取得最高信心度的類別名稱
        top1_id = results[0].probs.top1
        class_name = results[0].names[top1_id]
        confidence = float(results[0].probs.top1conf) * 100
        
        st.metric(label="辨識結果類別", value=class_name)
        st.write(f"**辨識信心度**：`{confidence:.2f}%`")
# 印出所有類別的信心度分配
st.write("各類別預測機率：", results[0].probs.data.tolist())
st.write("模型類別對照表：", results[0].names)
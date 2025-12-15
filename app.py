import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import sys

# 添加 src 到路徑以便導入模組
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from beautiful_photo import SignalProcessingAnalyzer, MathGuidedFilter, MediaPipeAnalyzer

# 設定頁面配置
st.set_page_config(
    page_title="多模態美顏相機 🌸",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂 CSS 美化介面
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF69B4;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF1493;
    }
    h1 {
        color: #FF69B4;
        text-align: center;
    }
    .upload-text {
        text-align: center;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

# 標題
st.title("📸 多模態美顏相機")
st.markdown("<p class='upload-text'>使用多模態處理技術為您的照片添加專業美顏效果</p>", unsafe_allow_html=True)

# 側邊欄 - 參數設定
st.sidebar.header("⚙️ 參數設定")

# 處理模式選擇
mode = st.sidebar.radio(
    "處理模式",
    ["自動偵測", "輕量模式", "強力模式"],
    help="自動偵測會根據照片狀況自動選擇；輕量保留較多質感；強力模式磨皮效果更明顯"
)

st.sidebar.markdown("---")

# 磨皮參數
st.sidebar.subheader("🎨 磨皮效果")
if mode == "輕量模式":
    default_r = 15
    default_eps = 0.05
elif mode == "強力模式":
    default_r = 25
    default_eps = 0.15
else:
    default_r = 15
    default_eps = 0.05

r = st.sidebar.slider(
    "磨皮半徑 (r)",
    min_value=5,
    max_value=50,
    value=default_r,
    step=5,
    help="數值越大磨皮效果越強，但可能失去細節"
)

eps = st.sidebar.slider(
    "細節保留 (eps)",
    min_value=0.01,
    max_value=0.3,
    value=default_eps,
    step=0.01,
    help="數值越小保留越多細節；越大磨皮越平滑"
)

st.sidebar.markdown("---")

# 美白與打光
st.sidebar.subheader("✨ 美化效果")
whitening = st.sidebar.slider(
    "美白強度",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="提亮皮膚色調"
)

brightness = st.sidebar.slider(
    "打光強度",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05,
    help="整體提升亮度"
)

st.sidebar.markdown("---")

# 瑕疵修復選項（僅強力模式）
if mode == "強力模式":
    blemish_repair = st.sidebar.checkbox(
        "啟用瑕疵修復",
        value=True,
        help="針對痘痘、紅點進行智能修復"
    )
else:
    blemish_repair = False

# 分析器選擇
analyzer_type = st.sidebar.selectbox(
    "分析引擎",
    ["MediaPipe (推薦)", "訊號處理"],
    help="MediaPipe 更快速準確；訊號處理為數學方法"
)

# 主要內容區域
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 上傳照片")
    uploaded_file = st.file_uploader(
        "選擇一張人像照片",
        type=['jpg', 'jpeg', 'png'],
        help="支援 JPG、JPEG、PNG 格式"
    )
    
    if uploaded_file is not None:
        # 顯示原始照片
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="原始照片", use_container_width=True)
        
        # 顯示照片資訊
        st.info(f"📊 圖片尺寸: {original_image.size[0]} x {original_image.size[1]} px")

with col2:
    st.subheader("✨ 處理結果")
    
    if uploaded_file is not None:
        # 處理按鈕
        if st.button("🎨 開始美顏處理", use_container_width=True):
            with st.spinner("正在處理中，請稍候..."):
                try:
                    # 儲存暫存檔案
                    temp_input = "temp_input.jpg"
                    original_image.save(temp_input)
                    
                    # 初始化分析器和濾鏡
                    if analyzer_type == "MediaPipe (推薦)":
                        analyzer = MediaPipeAnalyzer()
                    else:
                        analyzer = SignalProcessingAnalyzer()
                    
                    filter_tool = MathGuidedFilter()
                    
                    # 進度條
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 分析階段
                    status_text.text("🔍 正在分析照片...")
                    progress_bar.progress(30)
                    
                    protect_mask, acne_mask, score = analyzer.analyze_pipeline(temp_input)
                    
                    # 判斷處理模式
                    status_text.text("🎨 正在進行美顏處理...")
                    progress_bar.progress(60)
                    
                    THRESHOLD = 0.002
                    
                    if mode == "自動偵測":
                        # 自動判斷
                        if score < THRESHOLD:
                            st.info("🌟 診斷：膚況良好 → 採用輕量模式")
                            use_blemish = False
                        else:
                            st.warning("💪 診斷：瑕疵較多 → 採用強力模式")
                            use_blemish = True
                    elif mode == "輕量模式":
                        use_blemish = False
                    else:
                        use_blemish = blemish_repair
                    
                    # 處理影像
                    result_array = filter_tool.process_image(
                        temp_input,
                        mask=protect_mask,
                        blemish_mask=acne_mask if use_blemish else None,
                        r=r,
                        eps=eps,
                        whitening=whitening,
                        brightness=brightness
                    )
                    
                    # 將 numpy array 轉換為 PIL Image
                    result = Image.fromarray(result_array)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ 處理完成！")
                    
                    # 顯示結果
                    st.image(result, caption="處理後照片", use_container_width=True)
                    
                    # 儲存 session state 以便下載
                    st.session_state['processed_image'] = result
                    
                    # 清理暫存檔
                    if os.path.exists(temp_input):
                        os.remove(temp_input)
                    
                    st.success("🎉 美顏處理完成！")
                    
                except Exception as e:
                    st.error(f"❌ 處理過程發生錯誤: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # 下載按鈕
        if 'processed_image' in st.session_state:
            st.markdown("---")
            
            # 轉換為可下載的格式
            result_pil = st.session_state['processed_image']
            buf = io.BytesIO()
            result_pil.save(buf, format='JPEG', quality=95)
            byte_im = buf.getvalue()
            
            st.download_button(
                label="💾 下載處理後照片",
                data=byte_im,
                file_name="beautified_photo.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
    else:
        st.info("👈 請先上傳一張照片")

# 頁尾說明
st.markdown("---")
st.markdown("""
### 📖 使用說明
1. **上傳照片**：點擊左側上傳框選擇人像照片
2. **調整參數**：在左側邊欄設定處理參數
3. **開始處理**：點擊「開始美顏處理」按鈕
4. **下載結果**：處理完成後可下載美化後的照片

### 🎯 參數建議
- **輕量模式**：適合膚況較好的照片，保留更多自然質感
- **強力模式**：適合需要重點修飾的照片，磨皮效果更明顯
- **美白強度**：建議 0.2-0.4，過高會失去自然感
- **打光強度**：建議 0.1-0.2，輕微提亮即可

### 💡 技術說明
本應用採用多模態 AI 技術：
- **MediaPipe**：Google 開發的即時人臉檢測
- **頻率分離**：專業修圖技術，分離皮膚紋理與瑕疵
- **Guided Filter**：邊緣保留濾鏡，精準磨皮同時保留細節
""")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #999;'>多模態美顏相機 v1.0 | Powered by MediaPipe & Python</p>", unsafe_allow_html=True)

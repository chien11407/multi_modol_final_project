from PIL import Image
import matplotlib.pyplot as plt

# 引用你寫好的模組
from beautiful_photo import SignalProcessingAnalyzer, MathGuidedFilter

# main.py
def main():
    image_path = "./image.png"
    
    analyzer = SignalProcessingAnalyzer()
    filter_tool = MathGuidedFilter()

    # 1. 分析 (取得兩個 mask)
    protect_mask, blemish_mask = analyzer.analyze_pipeline(image_path)
    
    # 2. 處理 (傳入兩個 mask)
    result = filter_tool.process_image(
        image_path, 
        mask=protect_mask, 
        blemish_mask=blemish_mask, # 傳入痘痘遮罩
        r=25, 
        eps=0.02
    )
    
    show_comparison(image_path, result, protect_mask)
    

def show_comparison(original_path, result_arr, mask):
    """
    輔助函式：顯示 原圖 vs 遮罩 vs 結果圖
    """
    original = Image.open(original_path)
    
    plt.figure(figsize=(15, 5))
    
    # 1. 原圖
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis('off')
    
    # 2. 保護遮罩
    plt.subplot(1, 3, 2)
    plt.title("Protection Mask\n(White = Keep Detail)")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    # 3. 美顏結果
    plt.subplot(1, 3, 3)
    plt.title("Beauty Result\n(Guided Filter)")
    plt.imshow(result_arr)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
from PIL import Image
import matplotlib.pyplot as plt

# 引用你寫好的模組
from beautiful_photo import SignalProcessingAnalyzer, MathGuidedFilter

# main.py
def main():
    image_path = "./image_1.png"
    
    analyzer = SignalProcessingAnalyzer()
    filter_tool = MathGuidedFilter()

    # 1. 分析 (取得兩個 mask)
    protect_mask, acne_mask, score = analyzer.analyze_pipeline(image_path)
    
    # 2. 處理 (傳入兩個 mask)
    # 2. 設定判斷門檻 (0.2%)
    # 如果分數 > 0.002，代表滿臉痘痘，需要重手處理
    THRESHOLD = 0.01
    
    if score < THRESHOLD:
        print(">> 診斷：膚況不錯 (輕量模式)")
        # 輕量模式：acne_mask 傳入 None -> 不會跑中值濾波修復
        # 參數 r=15, eps=0.05 -> 保留較多皮膚質感
        result = filter_tool.process_image(
            image_path, 
            mask=protect_mask, 
            blemish_mask=None,    # <--- 關鍵：設為 None
            r=15, 
            eps=0.05
        )
    else:
        print(">> 診斷：瑕疵較多 (強力模式)")
        # 強力模式：acne_mask 傳入真的遮罩 -> 啟動中值濾波修復
        # 參數 r=25, eps=0.15 -> 磨得比較平，消除紅斑
        result = filter_tool.process_image(
            image_path, 
            mask=protect_mask, 
            blemish_mask=acne_mask, # <--- 關鍵：傳入遮罩
            r=25, 
            eps=0.1
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
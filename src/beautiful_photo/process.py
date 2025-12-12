import numpy as np
from PIL import Image

class ImageProcessor:
    def math_rgb_to_gray(img_array):
        """
        輸入: HxWx3 的 RGB 陣列
        輸出: HxW 的 灰階陣列
        原理: 矩陣內積 (Dot Product)
        """
        # 權重參數 (Luma coding in video systems)
        weights = np.array([0.299, 0.587, 0.114])
        
        # 利用 NumPy 的矩陣乘法，將 RGB 三個通道加權平均
        # img_array 的形狀是 (高, 寬, 3)，dot 運算會作用在最後一個維度
        gray_img = np.dot(img_array[...,:3], weights)
        
        # 轉回 0-255 的整數格式
        return gray_img.astype(np.uint8)

    def math_resize(img_array, target_h, target_w):
        """
        輸入: 影像陣列, 目標高度, 目標寬度
        原理: 最近鄰插值 (Nearest Neighbor Interpolation)
        """
        src_h, src_w = img_array.shape[:2]
        
        # 1. 建立目標影像的空白矩陣
        # 判斷是灰階(2維)還是彩色(3維)
        if len(img_array.shape) == 3:
            resized = np.zeros((target_h, target_w, img_array.shape[2]), dtype=np.uint8)
        else:
            resized = np.zeros((target_h, target_w), dtype=np.uint8)
        
        # 計算縮放比例 (Scale Factor)
        y_scale = src_h / target_h
        x_scale = src_w / target_w
        
        # 2. 進行座標映射 (Coordinate Mapping)
        # 對每一個目標像素 (y, x)，找出它對應回原圖的座標 (src_y, src_x)
        for y in range(target_h):
            for x in range(target_w):
                # 向下取整 (Floor) 找出最近的原始像素索引
                src_y = int(np.floor(y * y_scale))
                src_x = int(np.floor(x * x_scale))
                
                # 邊界檢查 (防止索引超出範圍)
                src_y = min(src_y, src_h - 1)
                src_x = min(src_x, src_w - 1)
                
                # 填值
                resized[y, x] = img_array[src_y, src_x]
                
        return resized

    # --- 執行流程：檔案轉換 ---
    def process_file_math(input_path, output_path):
        # 1. 讀取影像 (只使用 PIL 做 I/O，不使用其處理功能)
        img = Image.open(input_path)
        img_arr = np.array(img) # 轉為 NumPy 矩陣
        
        print(f"原始影像尺寸: {img_arr.shape}")

        # 2. 數學預處理：轉灰階
        gray_arr = math_rgb_to_gray(img_arr)
        
        # 3. 數學預處理：縮放到 300x300
        resized_arr = math_resize(gray_arr, 300, 300)
        
        # 4. 存檔
        output_img = Image.fromarray(resized_arr)
        output_img.save(output_path)
        print(f"處理完成 (數學法)，已儲存至: {output_path}")

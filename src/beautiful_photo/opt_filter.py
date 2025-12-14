import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class MathGuidedFilter:
    def __init__(self):
        pass

    def get_integral_image(self, img):
        """
        數學實作：積分圖 (Integral Image / Summed Area Table)
        利用累積加法 (Cumulative Sum) 快速建立查表
        """
        # 先沿著垂直方向累加，再沿著水平方向累加
        # 使用 float64 避免溢位
        integral = np.cumsum(img, axis=0).astype(np.float64)
        integral = np.cumsum(integral, axis=1)
        
        # 為了方便邊界計算，我們在上方和左方填充一行/列的 0
        h, w = img.shape
        padded_integral = np.zeros((h + 1, w + 1), dtype=np.float64)
        padded_integral[1:, 1:] = integral
        return padded_integral

    def box_filter_fast(self, img, r):
        """
        數學實作：O(1) 時間複雜度的 Box Filter
        利用積分圖原理：Sum = D + A - B - C
        """
        h, w = img.shape
        # 1. 計算積分圖
        S = self.get_integral_image(img)

        # 2. 定義矩形邊界
        # 我們要計算每個像素 (y, x) 為中心，半徑 r 的矩形平均
        # 矩形範圍：[y-r, y+r], [x-r, x+r]
        
        # 建立索引網格
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        
        # 計算四個頂點的座標 (注意要加上 padding 的位移 1)
        # 邊界限制 (Clip) 確保不超出圖像範圍
        y1 = np.clip(y_grid - r, 0, h)
        y2 = np.clip(y_grid + r + 1, 0, h)
        x1 = np.clip(x_grid - r, 0, w)
        x2 = np.clip(x_grid + r + 1, 0, w)

        # 3. 利用積分圖公式快速求和
        # Sum = S[y2, x2] - S[y1, x2] - S[y2, x1] + S[y1, x1]
        # 這裡利用 NumPy 的進階索引 (Advanced Indexing) 一次算完整張圖
        region_sum = (S[y2, x2] - S[y1, x2] - 
                      S[y2, x1] + S[y1, x1])

        # 4. 計算每個窗口內的像素數量 (Count)
        # 邊緣的窗口會比較小，所以要除以實際像素數，而不是 (2r+1)^2
        count = (y2 - y1) * (x2 - x1)
        
        return region_sum / count

    def guided_filter(self, I, p, r, eps):
        """
        導向濾波核心公式 (純 NumPy)
        I: 導向圖 (Guide)
        p: 輸入圖 (Input)
        r: 半徑
        eps: 正則化參數
        """
        print(f"正在執行數學導向濾波 (r={r}, eps={eps})...")
        
        # 1. 計算各種平均值 (Mean) - 使用我們手寫的 box_filter_fast
        mean_I  = self.box_filter_fast(I, r)
        mean_p  = self.box_filter_fast(p, r)
        mean_II = self.box_filter_fast(I * I, r)
        mean_Ip = self.box_filter_fast(I * p, r)

        # 2. 計算共變異數 (Covariance) 與 變異數 (Variance)
        # Cov(I, p) = E[Ip] - E[I]E[p]
        cov_Ip = mean_Ip - mean_I * mean_p
        # Var(I) = E[II] - E[I]^2
        var_I  = mean_II - mean_I * mean_I

        # 3. 計算線性係數 a, b (Linear Coefficients)
        # a = Cov(I, p) / (Var(I) + eps)
        a = cov_Ip / (var_I + eps)
        # b = Mean(p) - a * Mean(I)
        b = mean_p - a * mean_I

        # 4. 對係數 a, b 再做一次平滑
        mean_a = self.box_filter_fast(a, r)
        mean_b = self.box_filter_fast(b, r)

        # 5. 產生輸出
        q = mean_a * I + mean_b
        return q

    def process_image(self, image_path, mask=None, blemish_mask=None, r=15, eps=0.05):
        # 1. 讀取圖片
        img_pil = Image.open(image_path)
        w, h = img_pil.size
        img_arr = np.array(img_pil, dtype=np.float32) / 255.0
        
        # --- 步驟 0: 痘痘修復 (Blemish Inpainting) ---
        # 原理：利用積分圖計算局部平均，用「鄰域平均色」取代「痘痘色」
        if blemish_mask is not None:
            print("正在修復皮膚瑕疵 (使用積分圖 Box Filter)...")
            
            # A. 調整 Mask 大小 (Resize) - 使用 PIL
            b_mask_pil = Image.fromarray((blemish_mask * 255).astype(np.uint8))
            b_mask_resized = b_mask_pil.resize((w, h), Image.BILINEAR)
            # 轉回 Boolean Mask
            b_mask_arr = np.array(b_mask_resized) > 100 

            # B. 製作一張「全圖模糊」的底圖
            # 這裡直接呼叫我們自己寫的 box_filter_fast (純數學積分圖)
            # 半徑設為 5~8，足夠抹平痘痘即可
            inpainting_layer = np.zeros_like(img_arr)
            for i in range(3):
                inpainting_layer[:, :, i] = self.box_filter_fast(img_arr[:, :, i], r=10)

            # C. 替換像素
            # 如果是痘痘位置 (b_mask_arr 為 True)，就用模糊層的顏色取代原圖顏色
            # 這就是最簡單的「數位遮瑕膏」
            
            # 利用 NumPy Boolean Indexing 進行批次替換
            # img_arr[mask] = inpainting_layer[mask]
            # 需擴充 mask 維度以符合 RGB
            b_mask_3ch = np.dstack([b_mask_arr, b_mask_arr, b_mask_arr])
            
            # 執行替換
            img_arr = np.where(b_mask_3ch, inpainting_layer, img_arr)

        # --- 接下來繼續做導向濾波 (磨皮) ---
        smoothed = np.zeros_like(img_arr)
        for i in range(3):
            I_channel = img_arr[:, :, i]
            p_channel = img_arr[:, :, i]
            smoothed[:, :, i] = self.guided_filter(I_channel, p_channel, r, eps)
            
        # --- 遮罩合成 (Resize + Compose) ---
        if mask is not None:
            # 1. Resize Mask
            mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
            mask_pil = Image.fromarray(mask_uint8)
            mask_resized = mask_pil.resize((w, h), Image.BILINEAR)
            mask_float = np.array(mask_resized, dtype=np.float32) / 255.0
            
            # 2. 羽化 (使用積分圖 box_filter 代替高斯模糊)
            mask_soft = self.box_filter_fast(mask_float, r=2)
            
            # 3. 擴展維度
            mask_3ch = np.dstack([mask_soft, mask_soft, mask_soft])
            
            # 4. 合成
            final_output = img_arr * mask_3ch + smoothed * (1.0 - mask_3ch)
        else:
            final_output = smoothed

        return np.clip(final_output * 255, 0, 255).astype(np.uint8)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndimage

class MathGuidedFilter:
    def __init__(self):
        pass
    
    def get_integral_image(self, img):
        integral = np.cumsum(img, axis=0).astype(np.float64)
        integral = np.cumsum(integral, axis=1)
        
        # 為了方便邊界計算，我們在上方和左方填充一行/列的 0
        h, w = img.shape
        padded_integral = np.zeros((h + 1, w + 1), dtype=np.float64)
        padded_integral[1:, 1:] = integral
        return padded_integral

    def box_filter_fast(self, img, r):
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
    
    def process_image(self, image_path, mask=None, blemish_mask=None, r=15, eps=0.05, whitening=0.0, brightness=0.0):
        """
        使用頻率分離 (Frequency Separation) 技術進行人像磨皮
        這是一種接近專業修圖 (Retouching) 的方式，能保留皮膚質感 (Texture) 同時去除瑕疵 (Blemishes)。
        
        參數:
        - whitening: 美白強度 (0.0 ~ 1.0)，針對皮膚區域提亮
        - brightness: 整體亮度/打光 (0.0 ~ 1.0)，針對全圖提亮
        """
        print(f"正在處理影像 (頻率分離磨皮): {image_path}")
        print(f"參數設定: 磨皮(r={r}, eps={eps}), 美白={whitening}, 打光={brightness}")
        
        # 1. 讀取影像 (使用 PIL)
        try:
            pil_img = Image.open(image_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img_rgb = np.array(pil_img)
        except Exception as e:
            print(f"讀取影像發生錯誤: {e}")
            raise ValueError(f"無法讀取影像: {image_path}")

        # 獲取圖片尺寸
        h, w = img_rgb.shape[:2]
        print(f"影像尺寸: {w}x{h} (寬x高)")
        
        # 轉為浮點數 (0-1) 方便計算
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # ==========================================================
        # 步驟 0: 針對性紅點/痘痘修復 (Targeted Blemish Removal)
        # ==========================================================
        if blemish_mask is not None:
            # blemish_mask: 1=痘痘, 0=正常
            # 檢查是否有偵測到痘痘
            if np.sum(blemish_mask) > 0:
                print("正在執行針對性紅點修復 (Simple Inpainting)...")
                print(f"Blemish mask 原始尺寸: {blemish_mask.shape}")
                
                # 確保 blemish_mask 維度與影像一致
                if blemish_mask.shape[:2] != (h, w):
                    print(f"調整 blemish_mask 尺寸從 {blemish_mask.shape} 到 ({h}, {w})")
                    # 使用 scipy 的 zoom 調整大小
                    zoom_factors = (h / blemish_mask.shape[0], w / blemish_mask.shape[1])
                    blemish_mask = ndimage.zoom(blemish_mask, zoom_factors, order=0)  # order=0 nearest neighbor
                    print(f"調整後的 blemish_mask 尺寸: {blemish_mask.shape}")
                
                # 簡單修復：對痘痘區域使用高斯模糊填補 (或是使用周圍平均)
                # 對整張圖做一個較強的模糊
                
                # 分通道處理
                img_inpainted = np.zeros_like(img_float)
                for c in range(3):
                    channel = img_float[:, :, c]
                    # 使用高斯模糊作為填補來源
                    blurred_channel = ndimage.gaussian_filter(channel, sigma=3)
                    
                    # 在 mask 區域使用模糊後的值，其他區域保留原值
                    # 注意：blemish_mask 可能是 float (0.0-1.0) 或 bool
                    mask_c = blemish_mask > 0.5
                    channel_out = channel.copy()
                    channel_out[mask_c] = blurred_channel[mask_c]
                    img_inpainted[:, :, c] = channel_out
                
                # 更新 img_float
                img_float = img_inpainted
        
        # ==========================================================
        # 核心演算法：頻率分離 (Frequency Separation)
        # ==========================================================
        # 概念：影像 = 低頻 (膚色/光影) + 中頻 (瑕疵/痘痘) + 高頻 (毛孔/紋理)
        # 我們希望平滑「中頻」，但保留「低頻」和「高頻」。
        
        # 1. 提取高頻細節 (High Frequency / Texture)
        # 使用較小的 Gaussian Blur 來分離出非常細微的紋理 (如毛孔)
        # sigma=3 左右通常能涵蓋毛孔大小
        
        blur_small = np.zeros_like(img_float)
        for c in range(3):
            blur_small[:, :, c] = ndimage.gaussian_filter(img_float[:, :, c], sigma=3)
            
        high_freq_texture = img_float - blur_small
        
        # 2. 建立平滑基底 (Low Frequency / Base)
        # 原本使用 cv2.bilateralFilter，現在改用本類別實作的 Guided Filter
        # Guided Filter 也是一種 Edge-Preserving Smoothing Filter
        
        smooth_base = np.zeros_like(img_float)
        # Guided Filter 需要單通道處理
        # r 是半徑, eps 是正則化參數 (類似 sigmaColor 的作用)
        # 這裡 eps 需要是方差的尺度，通常 0.01~0.1 的平方
        eps_sq = eps * eps 
        
        for c in range(3):
            I = img_float[:, :, c]
            # 使用自身作為導向圖 (I, I)
            smooth_base[:, :, c] = self.guided_filter(I, I, r, eps_sq)
        
        # 3. 合成 (Reconstruct)
        # 結果 = 平滑基底 + 高頻紋理
        result = smooth_base + high_freq_texture
        
        # ==========================================================
        # 功能：美白 (Whitening) - 針對皮膚區域
        # ==========================================================
        if whitening > 0:
            # 使用 Gamma Correction 來提亮膚色 (Gamma < 1 會變亮)
            # 強度 0.0 -> Gamma 1.0 (不變)
            # 強度 1.0 -> Gamma 0.6 (顯著變亮)
            gamma = 1.0 - (whitening * 0.4)
            # 避免 gamma <= 0
            gamma = max(gamma, 0.1)
            
            # 只對 result (即將成為皮膚的部分) 做處理
            # 注意：這裡的 result 包含了高頻紋理，直接提亮可能會讓紋理變淡，
            # 但通常美白也會希望皮膚看起來通透一點，所以直接對 result 做 Gamma 是合理的。
            result = np.power(np.clip(result, 0.0, 1.0), gamma)

        # Clip 到合法範圍
        result = np.clip(result, 0.0, 1.0)
        
        # ==========================================================
        # 遮罩處理 (Masking)
        # ==========================================================
        if mask is not None:
            # mask: 1=保護(不磨皮), 0=皮膚(要磨皮)
            # 我們需要將 mask 調整大小並羽化
            
            print(f"Protect mask 原始尺寸: {mask.shape}")
            
            # 確保 mask 是 float32
            if mask.dtype != np.float32:
                mask = mask.astype(np.float32)
                
            # Resize mask to match image
            if mask.shape[:2] != (h, w):
                print(f"調整 protect mask 尺寸從 {mask.shape} 到 ({h}, {w})")
                # 使用 scipy.ndimage.zoom 進行縮放
                zoom_factors = (h / mask.shape[0], w / mask.shape[1])
                mask = ndimage.zoom(mask, zoom_factors, order=1)  # order=1 bilinear
                print(f"調整後的 protect mask 尺寸: {mask.shape}")
                
            # 羽化遮罩 (避免邊界生硬)
            mask_soft = ndimage.gaussian_filter(mask, sigma=5)
            
            # 擴展維度以符合 RGB
            if len(mask_soft.shape) == 2:
                mask_3ch = np.dstack([mask_soft, mask_soft, mask_soft])
            else:
                mask_3ch = mask_soft
                
            # 混合： 原圖 * 保護遮罩 + 磨皮(含美白)圖 * (1 - 保護遮罩)
            final_output = img_float * mask_3ch + result * (1.0 - mask_3ch)
        else:
            final_output = result

        # ==========================================================
        # 功能：亮度調整/打光 (Brightness) - 全局
        # ==========================================================
        if brightness > 0:
            # 簡單的線性增亮
            # 強度 0.0 -> +0.0
            # 強度 1.0 -> +0.2 (增加 20% 亮度)
            brightness_offset = brightness * 0.2
            final_output = final_output + brightness_offset

        # 轉回 uint8
        final_output_uint8 = (np.clip(final_output, 0, 1) * 255).astype(np.uint8)
        
        # 這裡直接回傳 RGB (因為我們是用 PIL 讀取的 RGB)
        return final_output_uint8
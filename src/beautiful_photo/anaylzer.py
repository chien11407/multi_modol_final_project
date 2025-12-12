import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
from scipy import ndimage

class SignalProcessingAnalyzer:
    def __init__(self):
        print("初始化訊號處理分析儀...")

    # ==========================================================
    # 第一階段：光照處理 (Illumination) - YUV & 直方圖等化
    # ==========================================================
    def rgb2yuv(self, rgb):
        """
        數學原理：線性變換 (Linear Transformation)
        Y (Luma) = 亮度, U/V (Chroma) = 色度
        """
        m = np.array([[ 0.29900, -0.16874,  0.50000],
                      [ 0.58700, -0.33126, -0.41869],
                      [ 0.11400,  0.50000, -0.08131]])
        yuv = np.dot(rgb, m)
        yuv[:,:,1:] += 128.0 # Offset UV to be positive
        return yuv

    def yuv2rgb(self, yuv):
        m = np.array([[ 1.00000, 1.00000, 1.00000],
                      [-0.00004, -0.34414, 1.77200],
                      [ 1.40200, -0.71414, 0.00001]])
        rgb = np.dot(yuv - [0, 128, 128], m)
        return np.clip(rgb, 0, 255)

    def histogram_equalization(self, channel):
        """
        數學原理：累積分布函數 (CDF)
        將亮度分佈映射到均勻分佈，增強全域對比度
        """
        # 計算直方圖 (PDF)
        hist, bins = np.histogram(channel.flatten(), 256, density=True)
        # 計算累積分布 (CDF)
        cdf = hist.cumsum()
        cdf = 255 * cdf / cdf[-1] # 正規化到 0-255
        
        # 使用線性插值將原始像素映射到新值
        img_eq = np.interp(channel.flatten(), bins[:-1], cdf)
        return img_eq.reshape(channel.shape)

    def gamma_correction(self, channel, gamma=1.0):
        """
        數學原理：冪次變換 (Power-Law)
        S = c * r^gamma
        """
        # 先正規化到 0-1
        norm = channel / 255.0
        corrected = np.power(norm, gamma)
        return corrected * 255.0

    # ==========================================================
    # 第二階段：邊緣感知 (Edge Perception) - Sobel
    # ==========================================================
    def sobel_gradients(self, img_gray):
        """
        數學原理：離散微分算子 (Discrete Differentiation Operator)
        計算水平與垂直方向的梯度
        """
        # Sobel Kernels
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        
        # 卷積 (Convolution)
        Ix = signal.convolve2d(img_gray, Kx, mode='same', boundary='symm')
        Iy = signal.convolve2d(img_gray, Ky, mode='same', boundary='symm')
        
        # 梯度強度 (Magnitude) G = sqrt(Ix^2 + Iy^2)
        magnitude = np.sqrt(Ix**2 + Iy**2)
        return magnitude / np.max(magnitude) # Normalize 0-1

    # ==========================================================
    # 第三階段：五官特徵提取 (Filter Banks) - Gabor Filters
    # ==========================================================
    def gabor_filter_bank(self, img_gray):
        """
        數學原理：Gabor 濾波器 (時頻分析)
        模擬人類視皮層 (V1) 對特定方向和頻率的響應。
        五官 (眼睛、嘴巴) 包含大量特定方向的高頻紋理。
        """
        print("正在計算 Gabor 濾波器組響應...")
        
        # 參數設定
        ksize = 15    # Kernel size
        sigma = 3.0   # 高斯包絡的標準差
        lambd = 8.0   # 正弦波波長 (決定頻率)
        gamma = 0.5   # 空間縱橫比
        
        # 累積所有方向的響應
        total_response = np.zeros_like(img_gray, dtype=np.float32)
        
        # 我們測試 4 個方向：0, 45, 90, 135 度
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for theta in thetas:
            # --- 手動建構 Gabor Kernel (數學公式) ---
            # G(x,y) = exp(-(x'^2 + y'^2*gamma^2)/(2*sigma^2)) * cos(2*pi*x'/lambd)
            # x' = x cos(theta) + y sin(theta)
            # y' = -x sin(theta) + y cos(theta)
            
            kernel = np.zeros((ksize, ksize))
            center = ksize // 2
            
            for y in range(ksize):
                for x in range(ksize):
                    px = x - center
                    py = y - center
                    
                    x_prime = px * np.cos(theta) + py * np.sin(theta)
                    y_prime = -px * np.sin(theta) + py * np.cos(theta)
                    
                    gaussian = np.exp(-(x_prime**2 + gamma**2 * y_prime**2) / (2 * sigma**2))
                    sinusoid = np.cos(2 * np.pi * x_prime / lambd)
                    kernel[y, x] = gaussian * sinusoid
            
            # 對影像進行濾波 (卷積)
            response = signal.convolve2d(img_gray, kernel, mode='same')
            total_response += response ** 2 # 能量疊加
            
        return total_response

    # ==========================================================
    # 主流程分析
    # ==========================================================
    def analyze_pipeline(self, image_path):
        # 0. 讀檔
        img_pil = Image.open(image_path).resize((400, 400)) # Resize 加速運算
        img_arr = np.array(img_pil, dtype=np.float32)
        
        # --- Stage 1: 光照處理 (YUV + HEQ) ---
        yuv = self.rgb2yuv(img_arr)
        Y_channel = yuv[:,:,0]
        
        # 先做 Gamma 校正 (防止過暗)
        Y_gamma = self.gamma_correction(Y_channel, gamma=0.9)
        # 再做直方圖等化 (增強對比)
        Y_eq = self.histogram_equalization(Y_gamma)
        
        # 更新 YUV 並轉回 RGB (這是光照校正後的圖)
        yuv[:,:,0] = Y_eq
        img_enhanced = self.yuv2rgb(yuv)
        
        # --- Stage 2: 邊緣感知 (Sobel) ---
        # 使用校正後的亮度 Y_eq 進行計算
        edges = self.sobel_gradients(Y_eq)
        
        # --- Stage 3: 特徵分析 (Gabor Filter Banks) ---
        # 五官通常具有強烈的紋理響應
        features_texture = self.gabor_filter_bank(Y_eq)
        features_texture = features_texture / np.max(features_texture) # Normalize
        
        # --- 綜合分析：產生遮罩 ---
        # 邊緣強 (輪廓) OR 紋理強 (五官內部細節) = 保護區
        # 閾值需要根據圖像調整，這裡設為經驗值
        mask_edge = edges > 0.15
        mask_feat = features_texture > 0.1
        
        # 最終保護遮罩 (白色為不可平滑區)
        final_mask = np.logical_or(mask_edge, mask_feat)
        
        # --- 視覺化 ---
        self.visualize(img_arr/255.0, img_enhanced/255.0, edges, features_texture, final_mask)
        
        return final_mask

    def visualize(self, original, enhanced, edges, gabor, mask):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0,0].imshow(original)
        axes[0,0].set_title("1. Original Image")
        
        axes[0,1].imshow(enhanced)
        axes[0,1].set_title("2. Illumination Corrected\n(YUV -> HEQ -> RGB)")
        
        axes[0,2].imshow(edges, cmap='gray')
        axes[0,2].set_title("3. Structure: Sobel Edges")
        
        axes[1,0].imshow(gabor, cmap='magma')
        axes[1,0].set_title("4. Features: Gabor Response\n(Eyes/Mouth Texture)")
        
        axes[1,1].imshow(mask, cmap='gray')
        axes[1,1].set_title("5. Final Protection Mask")
        
        # 模擬平滑結果 (綠色代表被平滑的皮膚)
        overlay = enhanced.copy()
        smooth_area = ~mask
        overlay[smooth_area] = overlay[smooth_area] * 0.7 + np.array([0, 0.3, 0]) * 0.3
        
        axes[1,2].imshow(overlay)
        axes[1,2].set_title("Analysis Overlay\n(Dark/Green = Smoothable Skin)")
        
        plt.tight_layout()
        plt.show()

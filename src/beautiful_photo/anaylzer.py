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
    def detect_blemishes(self, yuv_img):
        """
        數學原理：形態學頂帽運算 (Top-Hat Transform)
        公式：TopHat(f) = f - Open(f)
             Open(f) = Dilate(Erode(f))  (先腐蝕再膨脹)
        """
        print("正在偵測皮膚瑕疵 (使用 SciPy 形態學)...")
        
        # 1. 取出 Cr 通道 (紅色色差)
        # 痘痘在 Cr 通道通常數值較高 (偏紅)
        cr = yuv_img[:, :, 2]
        
        # 2. 定義結構元素的大小 (Kernel Size)
        # 假設痘痘半徑約 4-6 像素
        structure_size = 10
        
        # 3. 執行「灰階開運算 (Grayscale Opening)」
        # 數學意義：這會移除掉所有「小於」結構元素的亮點，只保留平緩的背景
        background = ndimage.grey_opening(cr, size=(structure_size, structure_size))
        
        # 4. 頂帽運算 (Top-Hat)
        # 原始圖 (有痘痘) - 背景圖 (沒痘痘) = 只剩下痘痘
        top_hat = cr - background
        
        # 5. 閾值化 (Thresholding)
        # 找出差異大於特定值的點 (這裡是經驗值，可調整)
        threshold = 5.0 
        blemish_mask = top_hat > threshold
        
        return blemish_mask

    def analyze_pipeline(self, image_path):
        # 1. 讀取與光照處理
        img_pil = Image.open(image_path).resize((400, 400))
        img_arr = np.array(img_pil, dtype=np.float32)
        yuv = self.rgb2yuv(img_arr)
        # ... (YUV 增強邏輯) ...
        Y_eq = yuv[:,:,0] # 假設這是增強後的 Y
        
        # 2. 邊緣與紋理 (略)
        edges = self.sobel_gradients(Y_eq)
        features = self.gabor_filter_bank(Y_eq)
        
        # --- [重點] 加入瑕疵偵測 ---
        # 使用原始的 YUV (未經直方圖等化，保留原始色差)
        blemish_mask = self.detect_blemishes(yuv)
        
        # --- 綜合分析 ---
        mask_edge = edges > 0.15
        mask_feat = features > 0.1
        
        # 保護區 = (五官 OR 邊緣)
        protection_mask = np.logical_or(mask_edge, mask_feat)
        
        # 最終邏輯：
        # 如果某個點原本被認為是特徵(例如很紅)，但它同時也是痘痘(blemish)，
        # 那我們就「取消保護」，強迫它進入磨皮/修復流程。
        final_protect_mask = np.logical_and(protection_mask, ~blemish_mask)
        
        # 回傳兩個 Mask
        return final_protect_mask, blemish_mask
    
    def analyze_pipeline(self, image_path):
        # 0. 讀檔
        img_pil = Image.open(image_path).resize((400, 400))
        img_arr = np.array(img_pil, dtype=np.float32)
        
        # Stage 1: YUV
        yuv = self.rgb2yuv(img_arr)
        Y_channel = yuv[:,:,0]
        Y_gamma = self.gamma_correction(Y_channel, gamma=0.9)
        Y_eq = self.histogram_equalization(Y_gamma)
        
        # Stage 2: Sobel
        edges = self.sobel_gradients(Y_eq)
        
        # Stage 3: Gabor
        features_texture = self.gabor_filter_bank(Y_eq)
        features_texture = features_texture / np.max(features_texture)
        
        # --- Stage 4: 瑕疵偵測 ---
        # 傳入原始 YUV (未經直方圖等化的，因為等化會破壞色差關係)
        blemish_mask = self.detect_blemishes(yuv)
        
        # --- 綜合分析 ---
        mask_edge = edges > 0.15
        mask_feat = features_texture > 0.1
        
        # 1. 原始保護區 (五官 + 邊緣)
        protection_mask = np.logical_or(mask_edge, mask_feat)
        
        # 2. 痘痘排除邏輯 (Cr Channel Exclusion)
        # 原理：痘痘在 YCbCr 的 Cr (紅色差) 通道數值會異常高
        # 我們利用統計學 (Mean + Std) 來自動找出這些紅點
        
        # 取出 Cr 通道 (yuv 是你前面算出來的變數)
        cr_channel = yuv[:, :, 2] 
        mean_cr = np.mean(cr_channel)
        std_cr = np.std(cr_channel)
         # 定義：比平均紅度高出 1.2 倍標準差的區域 = 痘痘/紅斑
        # 這個 1.2 可以微調 (越小抓越多痘痘)
        is_acne = cr_channel > (mean_cr + 1.2 * std_cr)
        
        # 3. 從保護區中「挖掉」痘痘
        # 邏輯：保護區 AND (NOT 痘痘)
        # 這樣五官還是白的，但臉頰上的紅痘痘會變成黑的 (可磨皮)
        refined_mask = np.logical_and(protection_mask, ~is_acne)
        
        # 4. 遮罩形態學優化 (Morphological Smoothing)
        # 目的：去除遮罩上的雜訊噪點，並讓邊緣柔和
        
        # A. 閉運算 (Closing): 把斷裂的線條接起來，填補小洞
        # 使用 5x5 的矩陣作為結構元素
        structure = np.ones((5, 5)) 
        mask_closed = ndimage.binary_closing(refined_mask, structure=structure)
        
        # B. 高斯羽化 (Gaussian Feathering)
        # 讓遮罩從 0/1 的二值變成 0.0~1.0 的灰階
        # sigma=2.0 代表羽化邊緣的寬度
        final_protect_mask = ndimage.gaussian_filter(mask_closed.astype(np.float32), sigma=2.0)
        
        return final_protect_mask, blemish_mask
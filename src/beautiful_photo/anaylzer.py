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
    # 改進的痘痘偵測：先檢測臉部，再檢測痘痘
    # ==========================================================
    def detect_blemishes(self, yuv_img):
        """
        改進的痘痘檢測方法 - 先限定臉部區域
        
        核心邏輯：
        1. 先檢測臉部區域（膚色檢測 + 最大連通域）
        2. 在臉部內檢測紅點（發炎痘痘）
        3. 排除五官邊界（梯度膨脹）
        """
        print("正在偵測痘痘 (先檢測臉部版)...")
        
        h, w = yuv_img.shape[:2]
        y_channel = yuv_img[:, :, 0].astype(np.float32)
        cb = yuv_img[:, :, 1].astype(np.float32)
        cr = yuv_img[:, :, 2].astype(np.float32)
        
        # ============ Step 0: 臉部區域檢測（膚色檢測）============
        # 原理：人類皮膚在 YUV 空間有特定的 Cb/Cr 範圍
        # Cb (藍色差)：通常在 80-130
        # Cr (紅色差)：通常在 130-180
        
        print(f"  - YUV 統計：Y=[{np.min(y_channel):.0f},{np.max(y_channel):.0f}], " + 
              f"Cb=[{np.min(cb):.0f},{np.max(cb):.0f}], Cr=[{np.min(cr):.0f},{np.max(cr):.0f}]")
        
        # 膚色範圍檢測（根據經驗值）
        skin_mask = np.logical_and.reduce([
            cb >= 70,   # Cb 下限
            cb <= 140,  # Cb 上限
            cr >= 130,  # Cr 下限
            cr <= 180   # Cr 上限
        ])
        
        print(f"  - 膚色候選：{np.sum(skin_mask)} 個像素 ({np.sum(skin_mask)/skin_mask.size*100:.2f}%)")
        
        # 形態學閉運算：填補膚色區域內的小洞
        morph_struct = np.ones((7, 7))
        skin_mask = ndimage.binary_closing(skin_mask, structure=morph_struct, iterations=2)
        
        # 找出最大連通域 = 臉部主區域
        labeled_skin, num_regions = ndimage.label(skin_mask)
        if num_regions == 0:
            print("  - 警告：未檢測到膚色區域！")
            return np.zeros((h, w), dtype=bool)
        
        # 計算每個連通域的大小
        region_sizes = ndimage.sum(skin_mask, labeled_skin, index=np.arange(1, num_regions + 1))
        
        # 找出最大的連通域（臉部）
        largest_region_label = np.argmax(region_sizes) + 1
        face_mask = (labeled_skin == largest_region_label)
        
        # 輕微膨脹臉部遮罩，避免邊緣遺漏
        face_mask = ndimage.binary_dilation(face_mask, structure=np.ones((5, 5)))
        
        print(f"  - 臉部區域：{np.sum(face_mask)} 個像素 ({np.sum(face_mask)/face_mask.size*100:.2f}%)")
        
        # ============ Step 1: 在臉部內檢測紅點 ============
        # 痘痘是發炎區域，Cr 值會異常高
        # 但只在臉部區域內計算統計值
        cr_in_face = cr[face_mask]
        mean_cr = np.mean(cr_in_face)
        std_cr = np.std(cr_in_face)
        
        print(f"  - 臉部 Cr 統計：mean={mean_cr:.2f}, std={std_cr:.2f}")
        
        # 計算 z-score：找出異常高的紅值區域
        # 只檢測臉部區域內的紅點
        cr_score = (cr - mean_cr) / (std_cr + 1e-6)
        red_mask = np.logical_and(cr_score > 0.5, face_mask)  # 必須在臉部內 + 紅值高
        
        print(f"  - 臉部內紅點：{np.sum(red_mask)} 個像素 ({np.sum(red_mask)/red_mask.size*100:.2f}%)")
        
        # ============ Step 2: 邊界清晰度檢測 (Sobel 梯度) ============
        # 目的：排除五官邊界（嘴唇、眼睛邊界）
        # 邏輯：高梯度 = 邊界清晰 = 五官 → 排除
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        
        grad_y = signal.convolve2d(cr, Ky, mode='same', boundary='symm')
        grad_x = signal.convolve2d(cr, Kx, mode='same', boundary='symm')
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 只在臉部區域內計算梯度統計
        grad_in_face = grad_magnitude[face_mask]
        grad_mean = np.mean(grad_in_face)
        grad_std = np.std(grad_in_face)
        
        print(f"  - 臉部梯度統計：mean={grad_mean:.2f}, std={grad_std:.2f}")
        
        # 使用統計方法判斷邊界清晰度
        # 五官邊界通常梯度很高，是異常值
        # 但我們還要排除五官內部！
        # 策略：標記高梯度區域，然後膨脹它，覆蓋整個五官
        grad_zscore = (grad_magnitude - grad_mean) / (grad_std + 1e-6)
        clear_boundary = grad_zscore > 1.0  # 識別清晰邊界
        
        # 膨脹邊界，擴展到整個五官內部
        dilate_struct = np.ones((15, 15))  # 大幅膨脹
        facial_features = ndimage.binary_dilation(clear_boundary, structure=dilate_struct)
        
        # 但只在臉部區域內標記五官（避免膨脹到臉外）
        facial_features = np.logical_and(facial_features, face_mask)
        
        print(f"  - 清晰邊界：{np.sum(clear_boundary)} 個像素 ({np.sum(clear_boundary)/clear_boundary.size*100:.2f}%)")
        print(f"  - 膨脹後五官區域：{np.sum(facial_features)} 個像素 ({np.sum(facial_features)/facial_features.size*100:.2f}%)")
        
        # ============ Step 3: 合併約束 ============
        # 核心邏輯：
        #   1. 必須在臉部區域內
        #   2. 必須有紅點（發炎）
        #   3. 排除五官區域（膨脹後的邊界）
        
        # 痘痘 = 臉部內的紅點 - 五官區域
        final_acne_mask = np.logical_and(red_mask, ~facial_features)
        
        print(f"  - 合併後：{np.sum(final_acne_mask)} 個像素 ({np.sum(final_acne_mask)/final_acne_mask.size*100:.2f}%)")
        
        # ============ Step 4: 形態學優化 ============
        # 簡化形態學操作，避免過度過濾
        morph_struct = np.ones((2, 2))  # 縮小結構元素
        
        # 只做輕微的閉運算，連接接近的痘痘
        final_acne_mask = ndimage.binary_closing(final_acne_mask, structure=morph_struct, iterations=1)
        
        print(f"  - 閉運算後：{np.sum(final_acne_mask)} 個像素")
        
        # 面積過濾：移除過小和過大的區域
        labeled_array, num_features = ndimage.label(final_acne_mask)
        if num_features > 0:
            blob_sizes = ndimage.sum(final_acne_mask, labeled_array, 
                                     index=np.arange(1, num_features + 1))
            
            # 打印連通域大小分布
            print(f"  - 連通域數量：{num_features}")
            print(f"  - 連通域大小：min={np.min(blob_sizes)}, max={np.max(blob_sizes)}, median={np.median(blob_sizes):.0f}")
            
            # 只過濾超大區域（可能是整個嘴唇）
            total_pixels = h * w
            max_size = int(total_pixels * 0.20)  # 最大 10%，排除超大區域
            
            # 只保留不超過上限的區域
            size_valid = blob_sizes <= max_size
            removed_count = num_features - np.sum(size_valid)
            
            if np.any(size_valid):
                valid_labels = np.where(size_valid)[0] + 1
                final_acne_mask = np.isin(labeled_array, valid_labels)
                print(f"  - 保留 {len(valid_labels)} 個連通域，移除 {removed_count} 個超大區域")
            else:
                final_acne_mask = np.zeros_like(final_acne_mask, dtype=bool)
                print(f"  - 所有連通域都太大（全部過濾）")
        
        print(f"  - 找到 {np.sum(final_acne_mask)} 個痘痘像素")
        
        return final_acne_mask
    
    # ==========================================================
    # 主流程分析
    # ==========================================================    
    def analyze_pipeline(self, image_path):
        # 0. 讀檔
        img_pil = Image.open(image_path).resize((400, 400))
        img_arr = np.array(img_pil, dtype=np.float32)
        
        # Stage 1: YUV
        yuv = self.rgb2yuv(img_arr)
        Y_channel = yuv[:,:,0]
        Y_gamma = self.gamma_correction(Y_channel, gamma=0.9)
        Y_eq = self.histogram_equalization(Y_gamma)
        
        # Stage 2: Sobel - 邊緣檢測
        # 目的：保護眼睛、嘴巴等邊界清晰的五官區域
        # 提高閾值，只保護非常明顯的邊緣
        Y_blurred = ndimage.gaussian_filter(Y_eq, sigma=1.0)
        edges = self.sobel_gradients(Y_blurred)
        edge_mask = edges > 0.40  # 提高到 0.40，只保護最清晰的邊緣
        
        # Stage 3: Gabor - 紋理特徵
        # 目的：識別五官的複雜紋理（眼睛睫毛、唇紋等）
        # 大幅提高閾值，避免把整個臉都標記為高紋理
        features_texture = self.gabor_filter_bank(Y_eq)
        features_texture = features_texture / np.max(features_texture)
        mask_feat = features_texture > 0.35  # 提高到 0.35，只保護最突出的紋理區域
        
        # --- Stage 4: 改進的瑕疵偵測 ---
        # 新方法已經內部包含多層約束，檢測出的都是痘痘
        blemish_mask = self.detect_blemishes(yuv)
        
        # 直接使用痘痘遮罩，不再用邊緣二次過濾
        # （邊緣檢測已在 detect_blemishes 中的梯度約束裡）
        final_acne_mask = blemish_mask.copy()
        
        # --- Stage 5: 生成處理遮罩 ---
        # 新策略：不再使用"五官保護遮罩"，而是直接反轉痘痘遮罩
        # 保護區 = NOT 痘痘 = 整個臉 - 痘痘
        # 這樣痘痘區域（mask=0）會被處理，其他區域（mask=1）保持原樣
        
        # 先適度擴張痘痘區域（確保完全覆蓋）
        dilate_struct = np.ones((5, 5))
        expanded_acne = ndimage.binary_dilation(final_acne_mask, structure=dilate_struct)
        
        # 反轉：保護遮罩 = 非痘痘區域
        # 痘痘區域是 0（會被處理），其他是 1（保持原圖）
        protect_mask = ~expanded_acne
        
        # 羽化邊界
        final_protect_mask = ndimage.gaussian_filter(protect_mask.astype(np.float32), sigma=3.0)
        
        # 計算瑕疵分數（用於判斷是否啟動強力模式）
        acne_score = np.sum(final_acne_mask) / final_acne_mask.size
        print(f"痘痘分數: {acne_score:.5f}")
               
        return final_protect_mask, final_acne_mask, acne_score
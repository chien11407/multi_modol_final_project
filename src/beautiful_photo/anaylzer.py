import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy import signal
from scipy import ndimage

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("Warning: MediaPipe not found. MediaPipeAnalyzer will not work.")

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

class MediaPipeAnalyzer:
    def __init__(self):
        print("初始化 MediaPipe 分析儀...")
        if not HAS_MEDIAPIPE:
            raise RuntimeError("MediaPipe is not installed. Please install it to use MediaPipeAnalyzer.")
            
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def analyze_pipeline(self, image_path):
        """
        使用 MediaPipe 偵測臉部特徵
        回傳:
        1. protect_mask: 保護區域 (眼睛、嘴巴、眉毛、背景) 為 1，皮膚為 0
        2. acne_mask: 這裡暫時回傳 None 或全黑，因為 MediaPipe 主要偵測幾何特徵
        3. score: 膚況分數 (暫時回傳 0，讓 main.py 走輕量模式，或依需求調整)
        """
        print(f"正在使用 MediaPipe 分析影像: {image_path}")
        
        # 1. 讀取影像 (使用 PIL)
        try:
            pil_image = Image.open(image_path)
            # 確保是 RGB
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            image_rgb = np.array(pil_image)
        except Exception as e:
            print(f"讀取影像發生錯誤: {e}")
            raise ValueError(f"無法讀取影像: {image_path}")
        
        h, w = image_rgb.shape[:2]
        
        # 2. 執行 Face Mesh
        results = self.face_mesh.process(image_rgb)
        
        # 準備 mask (預設全白 = 全部保護/不處理)
        # 我們希望皮膚部分是 0 (要處理)，其他部分是 1 (保護)
        
        # 使用 PIL ImageDraw 來繪製多邊形
        mask_img = Image.new('L', (w, h), 255) # 255 = White (Protect)
        draw = ImageDraw.Draw(mask_img)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            def get_coords(indices):
                coords = []
                for idx in indices:
                    pt = face_landmarks.landmark[idx]
                    coords.append((int(pt.x * w), int(pt.y * h)))
                return coords # List of tuples for PIL

            # 定義特徵區域的索引 (MediaPipe Face Mesh Topology)
            
            # Face Oval
            face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            
            # 1. Face Oval -> 設為 0 (皮膚區域)
            # MediaPipe 的 Face Oval 索引是有順序的，可以直接畫多邊形
            face_oval_points = get_coords(face_oval_indices)
            draw.polygon(face_oval_points, fill=0)

            # 2. 保護區域 -> 設為 1 (255)
            # 眼睛
            left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
            right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
            
            # 眉毛
            left_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
            right_eyebrow_indices = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
            
            # 嘴巴
            lips_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

            features = [left_eye_indices, right_eye_indices, left_eyebrow_indices, right_eyebrow_indices, lips_indices]
            
            for indices in features:
                points = get_coords(indices)
                draw.polygon(points, fill=255)

            # 轉回 numpy array (0-1 float)
            protect_mask = np.array(mask_img, dtype=np.float32) / 255.0

            # ==========================================================
            # 新增：紅點/痘痘偵測 (Red Spot / Acne Detection) - NumPy 實作
            # ==========================================================
            # 1. 轉換到 YCrCb 色彩空間
            # Formula:
            # Y = 0.299R + 0.587G + 0.114B
            # Cr = (R - Y) * 0.713 + 128
            # Cb = (B - Y) * 0.564 + 128
            
            R = image_rgb[:,:,0].astype(np.float32)
            G = image_rgb[:,:,1].astype(np.float32)
            B = image_rgb[:,:,2].astype(np.float32)
            
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            Cr = (R - Y) * 0.713 + 128.0
            
            # 2. 提取皮膚區域 (利用 protect_mask == 0 的部分)
            skin_mask_bool = protect_mask < 0.5
            
            # 3. 計算皮膚區域的 Cr 平均值與標準差
            if np.sum(skin_mask_bool) > 0:
                cr_skin = Cr[skin_mask_bool]
                mean_cr = np.mean(cr_skin)
                std_cr = np.std(cr_skin)
                
                # 4. 設定閾值
                threshold = mean_cr + 1.5 * std_cr
                
                # 5. 產生二值化遮罩
                acne_candidates = (Cr > threshold)
                
                # 6. 限制在皮膚區域內
                acne_mask_raw = np.logical_and(acne_candidates, skin_mask_bool)
                
                # 7. 形態學運算：去除雜訊 (太小的點)
                # 使用 scipy.ndimage.binary_opening
                # kernel size 3x3 roughly equivalent to iterations=1
                structure = ndimage.generate_binary_structure(2, 1) # 3x3 cross
                acne_mask_clean = ndimage.binary_opening(acne_mask_raw, structure=structure)
                
                # 8. 稍微膨脹一點
                acne_mask_final = ndimage.binary_dilation(acne_mask_clean, structure=structure, iterations=2)
                
                # 轉為 float32
                acne_mask = acne_mask_final.astype(np.float32)
                
                # 計算分數
                acne_pixels = np.sum(acne_mask_final)
                skin_pixels = np.sum(skin_mask_bool)
                score = acne_pixels / skin_pixels
                
                print(f"偵測到痘痘區域佔比: {score:.4%}")
            else:
                acne_mask = np.zeros((h, w), dtype=np.float32)
                score = 0.0
            
        else:
            # 沒偵測到臉
            protect_mask = np.ones((h, w), dtype=np.float32)
            acne_mask = np.zeros((h, w), dtype=np.float32)
            score = 0.0
        
        return protect_mask, acne_mask, score
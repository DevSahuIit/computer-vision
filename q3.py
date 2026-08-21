import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def gaussian_lpf_spatial(image, sigma):
    ksize = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def hybrid_spatial(img1, img2, sigma_low, sigma_high, alpha=0.5, beta=0.5):
    # Ensure both images have identical dimensions
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)
        
    lpf1 = gaussian_lpf_spatial(img1, sigma_low)
    lpf2 = gaussian_lpf_spatial(img2, sigma_high)
    
    # Compute high-pass filter: HPF = original - LPF
    hpf2 = img2.astype(np.float32) - lpf2.astype(np.float32)
    
    # Combine low and high spatial frequencies
    hybrid = alpha * lpf1.astype(np.float32) + beta * hpf2
    return np.clip(hybrid, 0, 255).astype(np.uint8)

def hybrid_frequency(img1, img2, cutoff_low, cutoff_high):
    # Ensure both images have identical dimensions
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)
        
    h, w = img1.shape
    
    # 2D FFT and Frequency Shift
    F1 = np.fft.fftshift(np.fft.fft2(img1))
    F2 = np.fft.fftshift(np.fft.fft2(img2))
    
    # Filter Meshgrid
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Gaussian Low-Pass and High-Pass Filters in Frequency Domain
    LPF = np.exp(-(dist**2) / (2 * (cutoff_low**2)))
    HPF = 1.0 - np.exp(-(dist**2) / (2 * (cutoff_high**2)))
    
    # Apply filters
    F1_filtered = F1 * LPF
    F2_filtered = F2 * HPF
    
    # Inverse FFT
    img_low = np.real(np.fft.ifft2(np.fft.ifftshift(F1_filtered)))
    img_high = np.real(np.fft.ifft2(np.fft.ifftshift(F2_filtered)))
    
    hybrid = img_low + img_high
    return np.clip(hybrid, 0, 255).astype(np.uint8)

# Load images
img1 = cv2.imread('face1.png', cv2.IMREAD_GRAYSCALE)  # Replace with actual image path
img2 = cv2.imread('face2.png', cv2.IMREAD_GRAYSCALE)  # Replace with actual image path

# Fallback for code execution testing if image loading fails
if img1 is None: img1 = np.ones((256, 256), dtype=np.uint8) * 200
if img2 is None: img2 = np.ones((200, 300), dtype=np.uint8) * 50  # Intentionally different size

# Pre-align / Match dimensions before passing into functions
target_height, target_width = 512, 512
img1 = cv2.resize(img1, (target_width, target_height), interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, (target_width, target_height), interpolation=cv2.INTER_AREA)

# Compute hybrid image in Spatial Domain
hybrid_sp = hybrid_spatial(img1, img2, sigma_low=5, sigma_high=3)

# Compute hybrid image in Frequency Domain
hybrid_freq = hybrid_frequency(img1, img2, cutoff_low=15, cutoff_high=25)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(hybrid_sp, cmap='gray')
plt.title('Spatial Domain Hybrid Image')

plt.subplot(1, 2, 2)
plt.imshow(hybrid_freq, cmap='gray')
plt.title('Frequency Domain Hybrid Image')

plt.tight_layout()
plt.show()
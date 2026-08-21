import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import data


def add_salt_and_pepper_noise(image, prob=0.05):
    """Adds salt and pepper noise to an image."""
    noisy = np.copy(image)
    
    num_salt = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255
    
    # Pepper (black pixels)
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    
    return noisy

def add_gaussian_noise(image, mean=0, std=25):
    """Adds Gaussian noise to an image."""
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def custom_box_filter(image, kernel_size):
    """
    Implements a Box Filter from scratch using 2D convolution,
    without using cv2.blur or cv2.filter2D.
    """
    pad = kernel_size // 2
    # Pad image to handle borders
    padded_img = np.pad(image, pad, mode='reflect')
    result = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the region of interest
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.mean(region)
            
    return result.astype(np.uint8)


def apply_filters(noisy_img, k):
    """Applies all 4 required filters with a specific kernel size."""
    # 1. Box Filter (Custom implementation)
    box = custom_box_filter(noisy_img, k)
    
    # 2. Weighted Average Filter (Using a binomial/center-weighted kernel)
    weight_kernel = np.ones((k, k), np.float32)
    weight_kernel[k//2, k//2] = k * 2 # Heavier center
    weight_kernel = weight_kernel / np.sum(weight_kernel)
    weighted_avg = cv2.filter2D(noisy_img, -1, weight_kernel)
    
    # 3. Gaussian Filter
    gaussian = cv2.GaussianBlur(noisy_img, (k, k), 0)
    
    # 4. Median Filter
    median = cv2.medianBlur(noisy_img, k)
    
    return {"Box": box, "Weighted Avg": weighted_avg, "Gaussian": gaussian, "Median": median}

def calculate_mse(img1, img2):
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

def calculate_psnr(img1, img2):
    mse = calculate_mse(img1, img2)
    if mse == 0: return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# ==========================================
# 4. Mixed Noise & Region-Wise Smoothing
# ==========================================
def mixed_noise_processing(img_clean):
    """Handles tasks 6 and 7: Mixed noise and region-wise smoothing."""
    h, w = img_clean.shape
    mid = w // 2
    
    # Create mixed noise image
    mixed_noisy = np.zeros_like(img_clean)
    left_sp = add_salt_and_pepper_noise(img_clean[:, :mid], prob=0.05)
    right_gauss = add_gaussian_noise(img_clean[:, mid:], std=25)
    
    mixed_noisy[:, :mid] = left_sp
    mixed_noisy[:, mid:] = right_gauss
    

    k = 5
    region_smoothed = np.zeros_like(img_clean)
    region_smoothed[:, :mid] = cv2.medianBlur(mixed_noisy[:, :mid], k)
    region_smoothed[:, mid:] = cv2.GaussianBlur(mixed_noisy[:, mid:], (k, k), 0)
    
    return mixed_noisy, region_smoothed

if __name__ == "__main__":

    # Load images: Camera (fine details) and Coins (smooth regions)
    img_details = cv2.resize(data.camera(), (256, 256))
    img_smooth = cv2.resize(data.coins(), (256, 256))
    
    # Add required noise
    noisy_details = add_salt_and_pepper_noise(img_details, prob=0.05)
    noisy_smooth = add_gaussian_noise(img_smooth, std=25)
    
    kernel_sizes = [3, 5, 7]
    
    print("--- Evaluating Fine Details Image (Salt & Pepper Noise) ---")
    for k in kernel_sizes:
        filters = apply_filters(noisy_details, k)
        for name, f_img in filters.items():
            psnr = calculate_psnr(img_details, f_img)
            print(f"Kernel {k}x{k} | {name:15s} | PSNR: {psnr:.2f} dB")
            
    # Display Absolute Difference for one kernel size (Task 5)
    example_k = 5
    filters_5x5 = apply_filters(noisy_details, example_k)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Absolute Difference D(x,y) for 5x5 Filters", fontsize=14)
    
    for ax, (name, f_img) in zip(axes.ravel(), filters_5x5.items()):
        # D(x,y) = |I(x,y) - If(x,y)|
        diff = np.abs(noisy_details.astype(np.float32) - f_img.astype(np.float32))
        img_plot = ax.imshow(diff, cmap='gray')
        ax.set_title(name)
        ax.axis('off')
        fig.colorbar(img_plot, ax=ax, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.show()

    # Process Mixed Noise Image (Tasks 6 & 7)
    mixed_noisy, region_smoothed = mixed_noise_processing(img_details)
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(mixed_noisy, cmap='gray')
    ax1.set_title("Mixed Noise\n(Left: S&P, Right: Gaussian)")
    ax1.axis('off')
    
    ax2.imshow(region_smoothed, cmap='gray')
    ax2.set_title("Region-Wise Smoothed\n(Left: Median, Right: Gaussian)")
    ax2.axis('off')
    plt.show()


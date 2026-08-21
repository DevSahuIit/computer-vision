import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def custom_resize_nearest_neighbour(img, new_shape):
    """
    Resizes an image using nearest-neighbour interpolation without built-in functions.
    """
    h, w = img.shape
    new_h, new_w = new_shape
    row_scale = h / new_h
    col_scale = w / new_w
    
    resized = np.zeros((new_h, new_w), dtype=np.uint8)
    
    for i in range(new_h):
        for j in range(new_w):
            orig_i = int(min(i * row_scale, h - 1))
            orig_j = int(min(j * col_scale, w - 1))
            resized[i, j] = img[orig_i, orig_j]
            
    return resized



def calculate_mse(img1, img2):
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

def calculate_psnr(img1, img2):
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))



def process_and_evaluate(image_path):
    # Load original 512x512 grayscale image
    img_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_orig is None or img_orig.shape != (512, 512):
        raise ValueError("Please provide a valid 512x512 grayscale image.")

    downsample_sizes = [(256, 256), (128, 128)]
    results = []

    for size in downsample_sizes:
        print(f"\nProcessing Downsampling to {size[0]}x{size[1]}")
        
        # 1. Downsample
        img_downsampled = cv2.resize(img_orig, size, interpolation=cv2.INTER_AREA)
        
        # 2. Resize back to 512x512
        target_size = (512, 512)
        
        # a. Nearest-Neighbour (Custom)
        recon_nn = custom_resize_nearest_neighbour(img_downsampled, target_size)
        
        # b. Bilinear (Built-in)
        recon_bilinear = cv2.resize(img_downsampled, target_size, interpolation=cv2.INTER_LINEAR)
        
        # c. Bicubic (Built-in)
        recon_bicubic = cv2.resize(img_downsampled, target_size, interpolation=cv2.INTER_CUBIC)
        
        methods = {
            "Nearest-Neighbour": recon_nn,
            "Bilinear": recon_bilinear,
            "Bicubic": recon_bicubic
        }
        
        # 3, 4, 6. Compare, Calculate Metrics, and Store Results
        for method_name, recon_img in methods.items():
            mse = calculate_mse(img_orig, recon_img)
            psnr = calculate_psnr(img_orig, recon_img)
            
            abs_err = np.abs(img_orig.astype(np.float64) - recon_img.astype(np.float64))
            sq_err = (img_orig.astype(np.float64) - recon_img.astype(np.float64)) ** 2
            
            results.append({
                "Downsample": f"{size[0]}x{size[1]}",
                "Method": method_name,
                "MSE": mse,
                "PSNR": psnr,
                "Abs Error": abs_err,
                "Sq Error": sq_err,
                "Reconstructed": recon_img
            })
            
            print(f"  {method_name:17s} -> MSE: {mse:.2f} | PSNR: {psnr:.2f} dB")
            
    return img_orig, results


def display_maps(img_orig, results):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    
    fig.suptitle("Error Maps for 256x256 Downsampling", fontsize=18, y=0.98)
    
    res_256 = [r for r in results if r["Downsample"] == "256x256"]
    
    for idx, res in enumerate(res_256):
        axes[idx, 0].imshow(res["Reconstructed"], cmap='gray')
        axes[idx, 0].set_title(f"{res['Method']}\nReconstructed", fontsize=14, pad=12)
        axes[idx, 0].axis('off')
        
        abs_err = res["Abs Error"]
        vmax_abs = np.percentile(abs_err, 99.5) if np.max(abs_err) > 0 else 1
        
        img_abs = axes[idx, 1].imshow(abs_err, cmap='hot', vmin=0, vmax=vmax_abs)
        axes[idx, 1].set_title(f"{res['Method']}\nAbsolute Error", fontsize=14, pad=12)
        axes[idx, 1].axis('off')
        fig.colorbar(img_abs, ax=axes[idx, 1], fraction=0.046, pad=0.04)
        
        sq_err = res["Sq Error"]
        vmax_sq = np.percentile(sq_err, 99.5) if np.max(sq_err) > 0 else 1
        
        img_sq = axes[idx, 2].imshow(sq_err, cmap='hot', vmin=0, vmax=vmax_sq)
        axes[idx, 2].set_title(f"{res['Method']}\nSquared Error", fontsize=14, pad=12)
        axes[idx, 2].axis('off')
        fig.colorbar(img_sq, ax=axes[idx, 2], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.4, wspace=0.3)
    plt.show()

if __name__ == "__main__":
    IMAGE_PATH = "img.png"
    
    try:
        original_img, exp_results = process_and_evaluate(IMAGE_PATH)
        display_maps(original_img, exp_results)
  
    except Exception as e:
        print(f"Error: {e}")
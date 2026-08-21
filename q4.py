import cv2
import numpy as np
import matplotlib.pyplot as plt

# Custom Convolution implementation for First & Second Order Derivatives
def custom_convolve2d(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)
            
    return output

# Load image
img = cv2.imread('boundaries.png', cv2.IMREAD_GRAYSCALE) # Add your image path
if img is None: img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

# 1. First-Order Derivatives (Sobel Kernels)
kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

gx = custom_convolve2d(img, kernel_x)
gy = custom_convolve2d(img, kernel_y)

# 2. Gradient Magnitude and Direction
magnitude = np.sqrt(gx**2 + gy**2)
direction = np.arctan2(gy, gx)

# 3. Binary Edge Maps using Three Thresholds
thresh_low = (magnitude > 50).astype(np.uint8) * 255
thresh_mid = (magnitude > 100).astype(np.uint8) * 255
thresh_high = (magnitude > 180).astype(np.uint8) * 255

# 4. Second-Order Derivative (Laplacian)
kernel_laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
second_order = custom_convolve2d(img, kernel_laplacian)
second_order_edges = (np.abs(second_order) > 30).astype(np.uint8) * 255

# 5. LoG and Canny Edge Detection
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
log_response = cv2.Laplacian(blurred, cv2.CV_64F)
log_edges = (np.abs(log_response) > 10).astype(np.uint8) * 255

canny_edges = cv2.Canny(img, threshold1=50, threshold2=150)

# Visualizations
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1); plt.imshow(gx, cmap='gray'); plt.title('Gradient X')
plt.subplot(2, 3, 2); plt.imshow(gy, cmap='gray'); plt.title('Gradient Y')
plt.subplot(2, 3, 3); plt.imshow(magnitude, cmap='gray'); plt.title('Gradient Magnitude')
plt.subplot(2, 3, 4); plt.imshow(thresh_mid, cmap='gray'); plt.title('1st-Order Thresh (Mid)')
plt.subplot(2, 3, 5); plt.imshow(second_order_edges, cmap='gray'); plt.title('2nd-Order Laplacian')
plt.subplot(2, 3, 6); plt.imshow(canny_edges, cmap='gray'); plt.title('Canny Edge Detector')
plt.tight_layout()
plt.show()
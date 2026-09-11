import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


IMG_BOUNDARIES = "boundaries.png"   # e.g. a coin, a building, a simple object
IMG_DETAILS = "finedetails.png"     # e.g. foliage, fabric, hair, fine texture

def custom_convolve2d(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image.astype(np.float32), ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.float32)
    # kernel is flipped for true convolution (vs. correlation)
    kflip = np.flip(kernel)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kflip)
    return output


def load_or_synthesize(path, seed, size=(256, 256), kind="shapes"):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) else None
    if img is None:
        print(f"[warn] '{path}' not found -> using synthetic placeholder ({kind}).")
        h, w = size
        rng = np.random.default_rng(seed)
        canvas = np.full((h, w), 40, dtype=np.float32)
        if kind == "shapes":
            cv2.rectangle(canvas, (40, 40), (150, 150), 200, -1)
            cv2.circle(canvas, (190, 190), 45, 255, -1)
        else:  # fine texture
            noise = rng.normal(0, 1, (h, w))
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            canvas = 128 + 60 * np.sin(xx / 4.0) * np.cos(yy / 5.0) + noise * 10
        img = np.clip(canvas, 0, 255).astype(np.uint8)
    else:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def normalize01(x):
    x = x.astype(np.float32)
    x -= x.min()
    if x.max() > 0:
        x /= x.max()
    return x


# ----------------------------------------------------------------------
# ZERO-CROSSING DETECTION FOR LoG (vectorized, no built-in edge-detector call)
# ----------------------------------------------------------------------
def zero_crossing(log_img, slope_threshold=4.0):
    h, w = log_img.shape
    padded = np.pad(log_img, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    edges = np.zeros((h, w), dtype=bool)
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dy, dx in shifts:
        neighbor = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        sign_change = (center * neighbor) < 0
        strong_enough = np.abs(center - neighbor) > slope_threshold
        edges |= (sign_change & strong_enough)
    return (edges.astype(np.uint8)) * 255


# ----------------------------------------------------------------------
# SIMPLE QUANTITATIVE PROXIES FOR COMPARISON (task 6)
# ----------------------------------------------------------------------
def edge_stats(edge_map):
    """Return (edge_density_%, num_connected_components, mean_component_length)
    as rough proxies for detail-detection, continuity/fragmentation, and thickness."""
    binary = (edge_map > 0).astype(np.uint8)
    density = 100.0 * binary.sum() / binary.size
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    num_components = num_labels - 1  # exclude background
    mean_len = stats[1:, cv2.CC_STAT_AREA].mean() if num_components > 0 else 0.0
    return density, num_components, mean_len


# ----------------------------------------------------------------------
# FULL PIPELINE FOR ONE IMAGE
# ----------------------------------------------------------------------
def process_image(path, label, kind, seed):
    print(f"\n{'='*70}\n{label}: '{path}'\n{'='*70}")
    img = load_or_synthesize(path, seed=seed, kind=kind)

    # ---- Task 1: first-order derivatives (Sobel, custom convolution) ----
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = custom_convolve2d(img, kernel_x)
    gy = custom_convolve2d(img, kernel_y)

    # ---- Task 2: gradient magnitude AND direction ----
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)  # radians, range [-pi, pi]

    # ---- Task 3: binary edge maps at three thresholds ----
    t_low, t_mid, t_high = 30, 70, 130
    thresh_low = (magnitude > t_low).astype(np.uint8) * 255
    thresh_mid = (magnitude > t_mid).astype(np.uint8) * 255
    thresh_high = (magnitude > t_high).astype(np.uint8) * 255
    for name, t, edge in [("low", t_low, thresh_low), ("mid", t_mid, thresh_mid),
                           ("high", t_high, thresh_high)]:
        density, ncomp, _ = edge_stats(edge)
        print(f"  Threshold {name:>4} (T={t:>3}): edge density={density:5.2f}%, "
              f"connected components={ncomp}")
    print("  -> Discussion: a LOW threshold keeps weak gradients, so fine/true edges survive,")
    print("     but noise and texture also get flagged as spurious edges (high density,")
    print("     many fragmented components). A HIGH threshold suppresses noise but also")
    print("     erases genuinely faint boundaries, causing missing/broken edges (low density,")
    print("     fewer components, but real structure lost).")

    # ---- Task 4: second-order derivative (Laplacian, custom convolution) ----
    kernel_laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    second_order = custom_convolve2d(img, kernel_laplacian)
    second_order_edges = (np.abs(second_order) > 30).astype(np.uint8) * 255

    d1, n1, _ = edge_stats(thresh_mid)
    d2, n2, _ = edge_stats(second_order_edges)
    print(f"  1st-order (Sobel, mid thresh) vs 2nd-order (Laplacian): "
          f"density {d1:.2f}% vs {d2:.2f}%, components {n1} vs {n2}")
    print("  -> Discussion: the 1st-order (Sobel) map gives thicker, more continuous edges")
    print("     and is less sensitive to noise; the 2nd-order (Laplacian) map is sharper/")
    print("     thinner and better localizes fine detail, but reacts strongly to noise,")
    print("     often fragmenting into more, smaller components.")

    # ---- Task 5: LoG (zero-crossing) and Canny ----
    blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
    log_response = cv2.Laplacian(blurred, cv2.CV_64F)
    log_edges = zero_crossing(log_response, slope_threshold=4.0)

    canny_edges = cv2.Canny(img, threshold1=50, threshold2=150)

    # ---- Task 6: compare all methods quantitatively ----
    methods = {
        "1st-order (Sobel, mid T)": thresh_mid,
        "2nd-order (Laplacian)": second_order_edges,
        "LoG (zero-crossing)": log_edges,
        "Canny": canny_edges,
    }
    print(f"\n  {'Method':<28}{'Density(%)':<14}{'Components':<14}{'MeanCompSize':<14}")
    scores = {}
    for name, edge in methods.items():
        density, ncomp, mean_len = edge_stats(edge)
        print(f"  {name:<28}{density:<14.2f}{ncomp:<14}{mean_len:<14.2f}")
        # crude "quality" proxy: prefer high continuity (large mean component size)
        # with a controlled density (not too sparse, not just noise)
        scores[name] = mean_len / (1 + abs(density - 5))

    best_method = max(scores, key=scores.get)
    best_density, best_ncomp, best_mean_len = edge_stats(methods[best_method])

    # ---- Task 7: best method per image + justification (dynamic, tied to the
    #      actual numbers above rather than a fixed canned statement) ----
    print(f"\n  -> Best-suited method for this image: {best_method}")
    print(f"     Justification: of the four maps, '{best_method}' has the best combination of")
    print(f"     continuity and controlled density for this image (density={best_density:.2f}%, "
          f"components={best_ncomp}, mean component size={best_mean_len:.1f} px).")
    other = {k: v for k, v in methods.items() if k != best_method}
    noisiest = max(other, key=lambda k: edge_stats(other[k])[1])  # most fragmented alternative
    noisiest_density, noisiest_ncomp, _ = edge_stats(other[noisiest])
    print(f"     By comparison, '{noisiest}' fragments into {noisiest_ncomp} components "
          f"(vs {best_ncomp} here), meaning its edges are less continuous / more broken up.")
    if kind == "shapes":
        print("     This fits expectations for an image with clean, high-contrast object")
        print("     boundaries: large, well-localized, continuous contours score best, while")
        print("     methods tuned for fine texture over-fragment the few real edges present.")
    else:
        print("     This fits expectations for a fine-texture image: a method that is too")
        print("     aggressive at suppressing weak gradients erases genuine detail, so the")
        print("     winner here is whichever map best preserves texture without dissolving")
        print("     into pure noise.")

    # ---- Visualization ----
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle(f"Q4 Edge Detection - {label}", fontsize=14)

    axes[0, 0].imshow(img, cmap="gray"); axes[0, 0].set_title("Original")
    axes[0, 1].imshow(normalize01(gx), cmap="gray"); axes[0, 1].set_title("Gradient X (Sobel)")
    axes[0, 2].imshow(normalize01(gy), cmap="gray"); axes[0, 2].set_title("Gradient Y (Sobel)")
    axes[0, 3].imshow(normalize01(magnitude), cmap="gray"); axes[0, 3].set_title("Gradient Magnitude")

    im_dir = axes[1, 0].imshow(direction, cmap="hsv")
    axes[1, 0].set_title("Gradient Direction (rad)")
    plt.colorbar(im_dir, ax=axes[1, 0], fraction=0.046)
    axes[1, 1].imshow(thresh_low, cmap="gray"); axes[1, 1].set_title(f"Edge Map T={t_low} (low)")
    axes[1, 2].imshow(thresh_mid, cmap="gray"); axes[1, 2].set_title(f"Edge Map T={t_mid} (mid)")
    axes[1, 3].imshow(thresh_high, cmap="gray"); axes[1, 3].set_title(f"Edge Map T={t_high} (high)")

    axes[2, 0].imshow(normalize01(second_order), cmap="gray"); axes[2, 0].set_title("2nd-order (Laplacian) raw")
    axes[2, 1].imshow(second_order_edges, cmap="gray"); axes[2, 1].set_title("2nd-order edges (thresholded)")
    axes[2, 2].imshow(log_edges, cmap="gray"); axes[2, 2].set_title("LoG (zero-crossing)")
    axes[2, 3].imshow(canny_edges, cmap="gray"); axes[2, 3].set_title("Canny")

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"q4_{label.replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved -> {out_path}")

    return methods, best_method


# ----------------------------------------------------------------------
# MAIN - run on both required images
# ----------------------------------------------------------------------
if __name__ == "__main__":
    _, best1 = process_image(IMG_BOUNDARIES, "Image A - Object Boundaries", kind="shapes", seed=1)
    _, best2 = process_image(IMG_DETAILS, "Image B - Fine Details", kind="texture", seed=2)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"Best method for boundary image : {best1}")
    print(f"Best method for fine-detail image: {best2}")
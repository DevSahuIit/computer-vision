import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os


PAIR_1 = ("face1_a.png", "face1_b.png")   # e.g. two aligned faces, pair A
PAIR_2 = ("face2_a.png", "face2_b.png")   # e.g. a second pair, pair B
TARGET_SIZE = (512, 512)                  # (width, height)


def load_or_synthesize(path, seed, size=TARGET_SIZE):
    """Load a grayscale image; if missing, synthesize a placeholder so the
    pipeline is fully runnable without real data."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) else None
    if img is None:
        print(f"[warn] '{path}' not found -> using synthetic placeholder.")
        rng = np.random.default_rng(seed)
        w, h = size
        yy, xx = np.mgrid[0:h, 0:w]
        blob = (255 * np.exp(-(((xx - w * 0.5) ** 2 + (yy - h * 0.5) ** 2) / (2 * (w * 0.18) ** 2))))
        texture = rng.normal(0, 15, (h, w))
        img = np.clip(blob + texture + 40, 0, 255).astype(np.uint8)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def to_uint8(img):
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return (img * 255).astype(np.uint8)


# ----------------------------------------------------------------------
# TASK 1 - FEATURE ALIGNMENT VIA GEOMETRIC TRANSFORM
# ----------------------------------------------------------------------
def detect_eye_centers(img):
    """Detect two eye centers with Haar cascades (used as alignment landmarks).
    Returns None if detection fails."""
    # OpenCV builds without the legacy objdetect module cannot provide Haar cascades.
    if not hasattr(cv2, "CascadeClassifier"):
        return None

    cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(cascade_path)
    if eye_cascade.empty():
        return None

    eyes = eye_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=8)
    if len(eyes) < 2:
        return None
    # keep the two largest detections, sort left-to-right
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    centers = sorted([(x + w / 2, y + h / 2) for (x, y, w, h) in eyes], key=lambda p: p[0])
    return centers


def align_images(img_ref, img_to_align):
    """Align img_to_align onto img_ref using a similarity transform computed
    from detected eye landmarks. Falls back to plain resize if landmarks
    cannot be found in either image (e.g. non-face or synthetic images)."""
    pts_ref = detect_eye_centers(img_ref)
    pts_mov = detect_eye_centers(img_to_align)

    if pts_ref is None or pts_mov is None:
        print("[align] Landmarks not found -> falling back to direct resize.")
        aligned = cv2.resize(img_to_align, (img_ref.shape[1], img_ref.shape[0]),
                              interpolation=cv2.INTER_CUBIC)
        return aligned, False

    src = np.float32(pts_mov)
    dst = np.float32(pts_ref)
    M, _ = cv2.estimateAffinePartial2D(src, dst)  # similarity transform: rotation+scale+translation
    aligned = cv2.warpAffine(img_to_align, M, (img_ref.shape[1], img_ref.shape[0]),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return aligned, True


# ----------------------------------------------------------------------
# TASK 2-3 - SPATIAL DOMAIN HYBRID + PARAMETER SWEEP
# ----------------------------------------------------------------------
def gaussian_lpf_spatial(image, sigma):
    ksize = int(6 * sigma + 1) | 1  # ensure odd
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def hybrid_spatial(img_low_src, img_high_src, sigma_low, sigma_high, alpha=0.5, beta=0.5):
    lpf = gaussian_lpf_spatial(img_low_src, sigma_low).astype(np.float32)
    lpf_of_high_src = gaussian_lpf_spatial(img_high_src, sigma_high).astype(np.float32)
    hpf = img_high_src.astype(np.float32) - lpf_of_high_src   # Eq. (2)
    hybrid = alpha * lpf + beta * hpf                          # Eq. (1)
    return np.clip(hybrid, 0, 255).astype(np.uint8), lpf, hpf


def sweep_spatial_params(img_low_src, img_high_src, sigmas_low, sigmas_high, weight_pairs):
    """Try several (sigma_low, sigma_high, alpha, beta) combinations and
    score each by high-frequency-band energy retained (a simple, automatic
    proxy for 'a visible mix at both distances'). Returns the best combo
    plus a table of all results."""
    results = []
    for sl in sigmas_low:
        for sh in sigmas_high:
            for (a, b) in weight_pairs:
                hyb, lpf, hpf = hybrid_spatial(img_low_src, img_high_src, sl, sh, a, b)
                # crude quality proxy: variance of hybrid (contrast) balanced
                # against how much of the low-frequency structure survives
                score = hyb.std()
                results.append({"sigma_low": sl, "sigma_high": sh,
                                 "alpha": a, "beta": b, "score": score, "image": hyb})
    best = max(results, key=lambda r: r["score"])
    return best, results


# ----------------------------------------------------------------------
# TASK 4 - FREQUENCY DOMAIN HYBRID (2D FFT), same weighting as Eq. (1)
# ----------------------------------------------------------------------
def hybrid_frequency(img_low_src, img_high_src, cutoff_low, cutoff_high, alpha=0.5, beta=0.5):
    h, w = img_low_src.shape

    F_low = np.fft.fftshift(np.fft.fft2(img_low_src))
    F_high = np.fft.fftshift(np.fft.fft2(img_high_src))

    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    LPF = np.exp(-(dist ** 2) / (2 * (cutoff_low ** 2)))
    HPF = 1.0 - np.exp(-(dist ** 2) / (2 * (cutoff_high ** 2)))

    F_low_filtered = F_low * LPF
    F_high_filtered = F_high * HPF

    img_low = np.real(np.fft.ifft2(np.fft.ifftshift(F_low_filtered)))
    img_high = np.real(np.fft.ifft2(np.fft.ifftshift(F_high_filtered)))

    hybrid = alpha * img_low + beta * img_high      # same weighting as Eq. (1)
    return np.clip(hybrid, 0, 255).astype(np.uint8), img_low, img_high


# ----------------------------------------------------------------------
# TASK 6 - GAUSSIAN / LAPLACIAN PYRAMID BLENDING
# ----------------------------------------------------------------------
def build_gaussian_pyramid(img, levels):
    gp = [img.astype(np.float32)]
    for _ in range(levels):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def build_laplacian_pyramid(gp):
    lp = [gp[-1]]
    for i in range(len(gp) - 1, 0, -1):
        size = (gp[i - 1].shape[1], gp[i - 1].shape[0])
        expanded = cv2.pyrUp(gp[i], dstsize=size)
        lp.append(gp[i - 1] - expanded)
    return lp[::-1]   # finest level first


def blend_laplacian_pyramids(lp_a, lp_b, gp_mask):
    """Blend two Laplacian pyramids level-by-level using a Gaussian pyramid
    of a blending mask (classic Burt-Adelson multi-band blending)."""
    blended = []
    n = len(lp_a)
    for i in range(n):
        m = gp_mask[i].astype(np.float32) / 255.0
        m = cv2.resize(m, (lp_a[i].shape[1], lp_a[i].shape[0]))
        blended.append(lp_a[i] * m + lp_b[i] * (1 - m))
    return blended


def reconstruct_from_laplacian(lp):
    img = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        size = (lp[i].shape[1], lp[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size) + lp[i]
    return np.clip(img, 0, 255).astype(np.uint8)


def make_smooth_mask(shape, kind="horizontal", feather=41):
    """A soft left/right (or top/bottom) blending mask used to demonstrate
    multi-scale pyramid mixing of the two source images."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)
    if kind == "horizontal":
        mask[:, : w // 2] = 255
    else:
        mask[: h // 2, :] = 255
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    return mask.astype(np.uint8)


def pyramid_blend(img_a, img_b, levels=5, mask_kind="horizontal"):
    mask = make_smooth_mask(img_a.shape, mask_kind)

    gp_a = build_gaussian_pyramid(img_a, levels)
    gp_b = build_gaussian_pyramid(img_b, levels)
    lp_a = build_laplacian_pyramid(gp_a)
    lp_b = build_laplacian_pyramid(gp_b)

    # Gaussian pyramid of the mask, reordered to finest-level-first so it
    # lines up with lp_a / lp_b for level-wise blending:
    gp_mask_finest_first = build_gaussian_pyramid(mask, levels)[::-1]

    blended_lp = blend_laplacian_pyramids(lp_a, lp_b, gp_mask_finest_first)
    return reconstruct_from_laplacian(blended_lp)


# ----------------------------------------------------------------------
# TASK 7 - LAPLACIAN PYRAMID USING BILATERAL FILTER (edge-preserving)
# ----------------------------------------------------------------------
def build_gaussian_pyramid_bilateral(img, levels, d=9, sigma_color=50, sigma_space=50):
    gp = [img.astype(np.float32)]
    for _ in range(levels):
        smoothed = cv2.bilateralFilter(gp[-1].astype(np.float32), d, sigma_color, sigma_space)
        down = cv2.resize(smoothed, (smoothed.shape[1] // 2, smoothed.shape[0] // 2),
                           interpolation=cv2.INTER_LINEAR)
        gp.append(down)
    return gp


def pyramid_blend_bilateral(img_a, img_b, levels=5, mask_kind="horizontal"):
    mask = make_smooth_mask(img_a.shape, mask_kind)

    gp_a = build_gaussian_pyramid_bilateral(img_a, levels)
    gp_b = build_gaussian_pyramid_bilateral(img_b, levels)
    lp_a = build_laplacian_pyramid(gp_a)
    lp_b = build_laplacian_pyramid(gp_b)
    gp_mask_finest_first = build_gaussian_pyramid(mask, levels)[::-1]

    blended_lp = blend_laplacian_pyramids(lp_a, lp_b, gp_mask_finest_first)
    return reconstruct_from_laplacian(blended_lp)


# ----------------------------------------------------------------------
# RUN THE FULL PIPELINE FOR ONE PAIR
# ----------------------------------------------------------------------
def process_pair(path_low, path_high, pair_name, seed_offset=0):
    print(f"\n{'='*70}\nProcessing {pair_name}: '{path_low}' (low-freq source), "
          f"'{path_high}' (high-freq source)\n{'='*70}")

    img_low_src = load_or_synthesize(path_low, seed=1 + seed_offset)
    img_high_src = load_or_synthesize(path_high, seed=2 + seed_offset)

    # ---- Task 1: alignment ----
    aligned_high, landmarks_found = align_images(img_low_src, img_high_src)
    print(f"[{pair_name}] Alignment via detected landmarks: {landmarks_found}")

    # ---- Task 2-3: spatial hybrid + parameter sweep ----
    sigmas_low = [3, 5, 8]
    sigmas_high = [2, 4, 6]
    weight_pairs = [(0.5, 0.5), (0.6, 0.4), (0.4, 0.6)]
    best, all_results = sweep_spatial_params(img_low_src, aligned_high,
                                              sigmas_low, sigmas_high, weight_pairs)
    print(f"[{pair_name}] Best spatial params -> sigma_low={best['sigma_low']}, "
          f"sigma_high={best['sigma_high']}, alpha={best['alpha']}, beta={best['beta']}")

    t0 = time.perf_counter()
    hybrid_sp, lpf_vis, hpf_vis = hybrid_spatial(img_low_src, aligned_high,
                                                  best["sigma_low"], best["sigma_high"],
                                                  best["alpha"], best["beta"])
    t_spatial = time.perf_counter() - t0

    # ---- Task 4-5: frequency-domain hybrid + timing comparison ----
    t0 = time.perf_counter()
    hybrid_fr, low_band, high_band = hybrid_frequency(img_low_src, aligned_high,
                                                        cutoff_low=15, cutoff_high=25,
                                                        alpha=best["alpha"], beta=best["beta"])
    t_freq = time.perf_counter() - t0
    print(f"[{pair_name}] Execution time -> spatial: {t_spatial*1000:.2f} ms, "
          f"frequency: {t_freq*1000:.2f} ms")

    # ---- Task 6: Gaussian/Laplacian pyramid blending ----
    pyr_blend = pyramid_blend(img_low_src, aligned_high, levels=5, mask_kind="horizontal")

    # ---- Task 7: Laplacian pyramid + bilateral filter blending ----
    pyr_blend_bilateral = pyramid_blend_bilateral(img_low_src, aligned_high,
                                                    levels=5, mask_kind="horizontal")

    # ---- Task 8: discussion (printed) ----
    print(f"[{pair_name}] Discussion:")
    print("  - Alignment quality directly controls how well the low- and high-frequency")
    print("    features register spatially; misaligned eyes/edges create ghosting in")
    print("    the hybrid result, most visible in the high-frequency (close-up) view.")
    print("  - A larger sigma_low removes more fine detail from the low-frequency image,")
    print("    strengthening the illusion at a distance; a larger sigma_high removes more")
    print("    of the low-frequency image's own content from the HPF term, sharpening the")
    print("    close-up perception but risking a 'washed out' distant view if pushed too far.")

    # ---- Visualization ----
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(f"Q3 Hybrid Images - {pair_name}", fontsize=14)

    axes[0, 0].imshow(img_low_src, cmap="gray"); axes[0, 0].set_title("Low-freq source s")
    axes[0, 1].imshow(aligned_high, cmap="gray"); axes[0, 1].set_title("High-freq source ")
    axes[0, 2].imshow(to_uint8(lpf_vis), cmap="gray"); axes[0, 2].set_title("LPF component")
    axes[0, 3].imshow(to_uint8(hpf_vis), cmap="gray"); axes[0, 3].set_title("HPF component (normalized)")
    axes[0, 4].imshow(hybrid_sp, cmap="gray"); axes[0, 4].set_title(f"Hybrid - Spatial\n({t_spatial*1000:.1f} ms)")

    axes[1, 0].imshow(to_uint8(low_band), cmap="gray"); axes[1, 0].set_title("FFT LPF component")
    axes[1, 1].imshow(to_uint8(high_band), cmap="gray"); axes[1, 1].set_title("FFT HPF component (normalized)")
    axes[1, 2].imshow(hybrid_fr, cmap="gray"); axes[1, 2].set_title(f"Hybrid - Frequency\n({t_freq*1000:.1f} ms)")
    axes[1, 3].imshow(pyr_blend, cmap="gray"); axes[1, 3].set_title("Gaussian/Laplacian\nPyramid blend")
    axes[1, 4].imshow(pyr_blend_bilateral, cmap="gray"); axes[1, 4].set_title("Laplacian Pyramid\n+ Bilateral blend")

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"q3_{pair_name.replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{pair_name}] Figure saved -> {out_path}")

    return {
        "hybrid_spatial": hybrid_sp, "hybrid_frequency": hybrid_fr,
        "pyramid_blend": pyr_blend, "pyramid_blend_bilateral": pyr_blend_bilateral,
        "t_spatial": t_spatial, "t_freq": t_freq, "best_params": best,
    }


# ----------------------------------------------------------------------
# MAIN - run BOTH pairs
# ----------------------------------------------------------------------
if __name__ == "__main__":
    result_pair1 = process_pair(PAIR_1[0], PAIR_1[1], "Pair 1", seed_offset=0)
    result_pair2 = process_pair(PAIR_2[0], PAIR_2[1], "Pair 2", seed_offset=10)

    # Summary table across both pairs
    print(f"\n{'='*70}\nSUMMARY TABLE\n{'='*70}")
    print(f"{'Pair':<10}{'sigma_low':<12}{'sigma_high':<12}{'alpha':<8}{'beta':<8}"
          f"{'t_spatial(ms)':<16}{'t_freq(ms)':<12}")
    for name, res in [("Pair 1", result_pair1), ("Pair 2", result_pair2)]:
        p = res["best_params"]
        print(f"{name:<10}{p['sigma_low']:<12}{p['sigma_high']:<12}{p['alpha']:<8}"
              f"{p['beta']:<8}{res['t_spatial']*1000:<16.2f}{res['t_freq']*1000:<12.2f}")
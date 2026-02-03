import time
import numpy as np
import cv2
from skimage import exposure
from skimage.transform import SimilarityTransform, warp
from skimage.registration import phase_cross_correlation
from skimage.metrics import normalized_mutual_information, structural_similarity
import SimpleITK as sitk
import pydicom


# -----------------------------
# UTILITIES
# -----------------------------
def dbg(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_dicom_gray(path):
    ds = pydicom.dcmread(path, force=True)
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    img -= img.min()
    img /= max(img.max(), 1e-8)
    return (img * 255).astype(np.uint8)


def force_2d(img):
    return img[0] if img.ndim == 3 else img


def normalize(img):
    img = exposure.equalize_adapthist(img, clip_limit=0.03)
    return (img * 255).astype(np.uint8)


def crop_black(img):
    mask = img > 5
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img, (0, img.shape[0], 0, img.shape[1])
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return img[y0 : y1 + 1, x0 : x1 + 1], (y0, y1 + 1, x0, x1 + 1)


def lift_transform(T_crop, crop_m, crop_f):
    """
    EXACT MATLAB lifting:
    Tf * T_crop * Tm
    """
    y0m, _, x0m, _ = crop_m
    y0f, _, x0f, _ = crop_f

    Tm = np.eye(3)
    Tm[:2, 2] = [-x0m, -y0m]

    Tf = np.eye(3)
    Tf[:2, 2] = [x0f, y0f]

    return Tf @ T_crop @ Tm


# -----------------------------
# SURF (MATLAB) EQUIVALENT
# -----------------------------
def try_surf(moving, fixed, min_matches=12):
    """
    MATLAB:
    detectSURFFeatures
    extractFeatures
    matchFeatures (MaxRatio=0.75)
    estimateGeometricTransform2D('similarity')
    """
    num_inliers = 0
    try:
        sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

        kp_m, desc_m = sift.detectAndCompute(moving, None)
        kp_f, desc_f = sift.detectAndCompute(fixed, None)

        if desc_m is None or desc_f is None:
            return None, False, 0

        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn = bf.knnMatch(desc_m, desc_f, k=2)

        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        if len(good) < min_matches:
            return None, False, 0

        src = np.float32([kp_m[m.queryIdx].pt for m in good])
        dst = np.float32([kp_f[m.trainIdx].pt for m in good])

        M, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=10,
            maxIters=3000,
            confidence=0.999,
        )
        num_inliers = int(inliers.sum())

        if M is None or inliers.sum() < 8:
            return None, False, num_inliers

        T = np.eye(3)
        T[:2, :] = M

        return T, True, num_inliers

    except Exception as e:
        dbg(f"SURF error: {e}")
        return None, False, num_inliers


# -----------------------------
# FALLBACKS (MATLAB-style)
# -----------------------------
def try_mutual_info(moving, fixed):
    m = sitk.GetImageFromArray(moving.astype(np.float32))
    f = sitk.GetImageFromArray(fixed.astype(np.float32))

    init = sitk.CenteredTransformInitializer(
        f,
        m,
        sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-6, 200)
    reg.SetInitialTransform(init, False)

    try:
        out = reg.Execute(f, m)
        mat = np.array(out.GetMatrix()).reshape(2, 2)
        tr = np.array(out.GetTranslation())

        T = np.eye(3)
        T[:2, :2] = mat
        T[:2, 2] = tr
        return T, True
    except:
        return None, False


# def try_phase_corr(moving, fixed):
#     shift, _, _ = phase_cross_correlation(fixed, moving)
#     T = np.eye(3)
#     T[:2, 2] = shift[::-1]
#     return T, True


def try_phase_corr(moving, fixed, min_size=32, max_error=0.25, max_shift_ratio=0.30):
    """
    Robust phase correlation with failure criteria.

    Returns:
        T (3x3) , True   → valid translation
        None, False     → reject result
    """

    h = min(moving.shape[0], fixed.shape[0])
    w = min(moving.shape[1], fixed.shape[1])

    # ---- Size sanity ----
    if h < min_size or w < min_size:
        dbg("PhaseCorr: image too small")
        return None, False

    # ---- Resize to common space ----
    moving_r = cv2.resize(moving, (w, h), interpolation=cv2.INTER_LINEAR)
    fixed_r = cv2.resize(fixed, (w, h), interpolation=cv2.INTER_LINEAR)

    # ---- Phase correlation ----
    shift, error, _ = phase_cross_correlation(fixed_r, moving_r, upsample_factor=1)

    # ---- Error gate (CRITICAL) ----
    if not np.isfinite(error) or error > max_error:
        dbg(f"PhaseCorr rejected: weak peak (error={error:.3f})")
        return None, False

    # ---- Scale shift back to original resolution ----
    sy = fixed.shape[0] / h
    sx = fixed.shape[1] / w

    dx = shift[1] * sx
    dy = shift[0] * sy

    # ---- Translation sanity ----
    max_dx = max_shift_ratio * fixed.shape[1]
    max_dy = max_shift_ratio * fixed.shape[0]

    if abs(dx) > max_dx or abs(dy) > max_dy:
        dbg(f"PhaseCorr rejected: excessive shift " f"(dx={dx:.1f}, dy={dy:.1f})")
        return None, False

    # ---- Build transform ----
    T = np.eye(3)
    T[:2, 2] = [dx, dy]

    dbg(f"PhaseCorr accepted: " f"dx={dx:.1f}, dy={dy:.1f}, error={error:.3f}")

    return T, True


def is_invertible(T, eps=1e-8):
    A = T[:2, :2]
    return abs(np.linalg.det(A)) > eps


# -----------------------------
# MAIN REGISTRATION
# -----------------------------
def register_custom(mp_full, op_full):
    dbg("Preprocessing")
    mp = normalize(mp_full)
    op = normalize(op_full)

    mp_c, crop_m = crop_black(mp)
    op_c, crop_f = crop_black(op)

    dbg("Trying SURF (MATLAB equivalent)")
    T, ok, inlier_matches = try_surf(mp_c, op_c)

    if not ok:
        dbg("SURF failed → Mutual Information")
        T, ok = try_mutual_info(mp_c, op_c)

    if not ok:
        dbg("MI failed → Phase Correlation")
        T, ok = try_phase_corr(mp_c, op_c)

    if not ok:
        raise RuntimeError("All registration methods failed")

    T = lift_transform(T, crop_m, crop_f)
    tform = SimilarityTransform(matrix=T)

    # Mapped images
    mp_mapped = warp(
        mp_full, tform.inverse, output_shape=op_full.shape, preserve_range=True
    ).astype(np.uint8)

    op_mapped = warp(
        op_full, tform, output_shape=mp_full.shape, preserve_range=True
    ).astype(np.uint8)

    return {
        "mp_original": mp_full,
        "op_original": op_full,
        "mp_mapped": mp_mapped,
        "op_mapped": op_mapped,
        "tform": tform,
        "crop_m": crop_m,
        "crop_f": crop_f,
        "num_features": inlier_matches,
    }


def save_registration_results(res_dict, out_dir="output"):
    import os

    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, "mp_original.png"), res_dict["mp_original"])
    cv2.imwrite(os.path.join(out_dir, "op_original.png"), res_dict["op_original"])
    cv2.imwrite(os.path.join(out_dir, "mp_mapped.png"), res_dict["mp_mapped"])
    cv2.imwrite(os.path.join(out_dir, "op_mapped.png"), res_dict["op_mapped"])

    # Save transform
    np.savetxt(
        os.path.join(out_dir, "transform.txt"), res_dict["tform"].params, fmt="%.6f"
    )


# -----------------------------
# METRICS
# -----------------------------
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def compute_metrics(fixed, warped, num_features=None):
    """
    Computes RMSE, NMI, and returns the number of mapped features.

    Parameters:
        fixed (np.ndarray): reference image
        warped (np.ndarray): registered / warped image
        num_features (int, optional): number of matched features from registration
    Returns:
        dict: { "NMI": float, "RMSE": float, "num_features": int }
    """
    fixed = fixed.astype(np.float32)
    warped = warped.astype(np.float32)

    # --- RMSE ---
    rmse = float(np.sqrt(np.mean((fixed - warped) ** 2)))

    # --- NMI ---
    f_flat = fixed.flatten()
    w_flat = warped.flatten()
    nmi = float(
        normalized_mutual_info_score(f_flat.astype(np.int32), w_flat.astype(np.int32))
    )

    return {
        "NMI": nmi,
        #    "RMSE": rmse,
        #    "num_features": num_features if num_features is not None else 0,
    }


def map_mp_to_op_custom(pt_mp, tform, crop_m, crop_f):
    y0_m, _, x0_m, _ = crop_m
    y0_f, _, x0_f, _ = crop_f

    r_crop = pt_mp[0] - y0_m
    c_crop = pt_mp[1] - x0_m

    x, y = float(c_crop), float(r_crop)
    xp, yp = tform([[x, y]])[0]

    r_full = yp + y0_f
    c_full = xp + x0_f
    return np.array([r_full, c_full])


def visualize_custom_point_mapping(mp, op, tform, crop_m, crop_f):
    mp_disp = cv2.cvtColor(mp, cv2.COLOR_GRAY2BGR)
    op_disp = cv2.cvtColor(op, cv2.COLOR_GRAY2BGR)

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        print(f"\nMP click: x={x}, y={y}")
        cv2.circle(mp_disp, (x, y), 4, (255, 0, 0), -1)

        mapped = map_mp_to_op_custom((y, x), tform, crop_m, crop_f)
        if not np.all(np.isfinite(mapped)):
            print("⚠️ Invalid transform")
            return

        r_op, c_op = mapped
        xi, yi = int(round(c_op)), int(round(r_op))
        print(f"CUSTOM mapped OP: x={xi}, y={yi}")

        if 0 <= xi < op.shape[1] and 0 <= yi < op.shape[0]:
            cv2.circle(op_disp, (xi, yi), 6, (0, 255, 0), -1)

    cv2.namedWindow("MP (Custom)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("OP (Custom)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MP (Custom)", 900, 900)
    cv2.resizeWindow("OP (Custom)", 900, 900)

    cv2.setMouseCallback("MP (Custom)", on_click)

    while True:
        cv2.imshow("MP (Custom)", mp_disp)
        cv2.imshow("OP (Custom)", op_disp)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


import os

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":

    dicom_mp = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\FAF\LBS-008-CT03-120101-IR-V5_converted\120101_203559_2451356_IR_OP_0.dcm"
    dicom_op = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\OCT\LBS-008-CT03-120101-OCT-V5_converted\120101_203559_2451341_fundus_0.dcm"

# if __name__ == "__main__":

#     faf_dir = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\FAF\LBS-008-CT03-120101-IR-V5_converted"
#     cfp_dir = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\OCT\LBS-008-CT03-120101-OCT-V5_converted"

#     faf_files = sorted(
#         [
#             os.path.join(faf_dir, f)
#             for f in os.listdir(faf_dir)
#             if f.lower().endswith(".dcm")
#         ]
#     )
#     cfp_files = sorted(
#         [
#             os.path.join(cfp_dir, f)
#             for f in os.listdir(cfp_dir)
#             if f.lower().endswith(".dcm")
#         ]
#     )

#     dbg(f"Found {len(faf_files)} FAF DICOMs")
#     dbg(f"Found {len(cfp_files)} CFP DICOMs")

#     for faf_path in faf_files:
#         for cfp_path in cfp_files:

#             dbg("=" * 60)
#             dbg(f"FAF: {os.path.basename(faf_path)}")
#             dbg(f"CFP: {os.path.basename(cfp_path)}")

#             try:
#                 mp = force_2d(load_dicom_gray(faf_path))
#                 op = force_2d(load_dicom_gray(cfp_path))

#                 res = register_custom(mp, op)

#                 metrics = compute_metrics(
#                     res["op_original"],
#                     res["mp_mapped"],
#                     num_features=res["num_features"],
#                 )

#                 dbg(
#                     f"SUCCESS | "
#                     f"NMI={metrics['NMI']:.4f} | "
#                     f"Features={res['num_features']}"
#                 )

#                 # ---- Per-pair output folder ----
#                 pair_name = (
#                     os.path.splitext(os.path.basename(faf_path))[0]
#                     + "__TO__"
#                     + os.path.splitext(os.path.basename(cfp_path))[0]
#                 )
#                 out_dir = os.path.join("output", pair_name)

#                 save_registration_results(res, out_dir=out_dir)

#             except Exception as e:
#                 dbg(f"FAILED: {e}")
#                 continue

mp = force_2d(load_dicom_gray(dicom_mp))
op = force_2d(load_dicom_gray(dicom_op))

res = register_custom(mp, op)

print("\nCUSTOM TRANSFORM MATRIX:")
print(res["tform"].params)

metrics = compute_metrics(
    res["op_original"], res["mp_mapped"], num_features=res["num_features"]
)

NMI_THRESH = 0.09
nmi = metrics.get("NMI", 0.0)

if not np.isfinite(nmi) or nmi < NMI_THRESH:
    raise RuntimeError(f"Registration FAILED (NMI={nmi:.4f})")

print("\n=== CUSTOM REGISTRATION ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Save all images
save_registration_results(res, out_dir="output")

# Optional visualization
visualize_custom_point_mapping(
    res["mp_original"],
    res["op_original"],
    res["tform"],
    res["crop_m"],
    res["crop_f"],
)

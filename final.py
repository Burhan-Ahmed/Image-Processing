import os
import time
import zipfile
import tempfile
import numpy as np
import cv2
import imageio.v2 as imageio
import requests
from scipy.optimize import least_squares
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# -----------------------------
# DEBUG
# -----------------------------
def dbg(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -----------------------------
# CONFIG
# -----------------------------
CONVERSION_API = "https://20.163.29.143:8001/api/v1/convert"
REGISTRATION_API = "http://20.163.29.143:8080/process"

CONVERSION_KEY = "dev-key-123"
REGISTRATION_KEY = (
    "sk-prod_C3YDJIml67G_o8LVm6eryB9bRZ9o2RYmuV_9PntFCZVR6cRnpLNxvEBgpsm2RkoT"
)


# -----------------------------
# HTTP SESSION (RETRIES + STREAMING)
# -----------------------------
def make_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


SESSION = make_session()


def stream_to_file(response, out_path, chunk=1024 * 1024):
    with open(out_path, "wb") as f:
        for c in response.iter_content(chunk_size=chunk):
            if c:
                f.write(c)


# -----------------------------
# STEP 2: REGISTER
# -----------------------------
def register_dicoms(mp_path, op_path):
    dbg("Registering MP and OP dicoms")

    headers = {"X-API-Key": REGISTRATION_KEY}

    with open(mp_path, "rb") as mp, open(op_path, "rb") as op:
        response = SESSION.post(
            REGISTRATION_API,
            headers=headers,
            params={"format": "zip"},
            files={"mp_dicom": mp, "op_dicom": op},
            stream=True,
            timeout=600,
        )

    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as zf:
        stream_to_file(response, zf.name)

    mp_img = op_img = transform_txt = None

    with zipfile.ZipFile(zf.name) as z:
        for n in z.namelist():
            if "mapped" not in n and "MP" in n:
                mp_img = tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(n)[1]
                ).name
                open(mp_img, "wb").write(z.read(n))
            elif "mapped" not in n and "OP" in n:
                op_img = tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(n)[1]
                ).name
                open(op_img, "wb").write(z.read(n))
            elif n.endswith(".txt"):
                transform_txt = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".txt"
                ).name
                open(transform_txt, "wb").write(z.read(n))

    if not all([mp_img, op_img, transform_txt]):
        raise RuntimeError("Missing output files from registration")

    return mp_img, op_img, transform_txt


# -----------------------------
# TRANSFORM PARSING
# -----------------------------
def parse_i2k_transform(txt_path):
    transforms = {}
    montage_origin = None

    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith("MONTAGE_ORIGIN"):
            _, ox, oy = lines[i].split()
            montage_origin = np.array([float(oy), float(ox)])
            print("Montage_origin:", montage_origin)
        elif lines[i].startswith("IMAGE_NAME"):
            name = lines[i].split()[-1]
            if i + 3 < len(lines) and lines[i + 1] == "QUADRATIC":
                A = np.fromstring(lines[i + 2], sep=" ")
                B = np.fromstring(lines[i + 3], sep=" ")
                transforms[name] = (A, B)
                # Optional: flag identity
                if np.allclose(A, [0, 0, 0, 1, 0, 0]) and np.allclose(
                    B, [0, 0, 0, 0, 1, 0]
                ):
                    print(f"Note: {name} is identity / affine-only")
                i += 3
        i += 1

    if montage_origin is None or not transforms:
        raise RuntimeError("Invalid transform file: no valid transforms found")

    return transforms, montage_origin


def quadratic_transform(X, Y, A, B):
    X2, Y2, XY = X * X, Y * Y, X * Y
    Xp = A[0] * X2 + A[1] * XY + A[2] * Y2 + A[3] * X + A[4] * Y + A[5]
    Yp = B[0] * X2 + B[1] * XY + B[2] * Y2 + B[3] * X + B[4] * Y + B[5]
    return Xp, Yp


def inverse_quadratic(Xp, Yp, A, B, x0=None):
    if x0 is None:
        x0 = [Xp, Yp]

    def residual(p):
        X, Y = p
        X2, Y2, XY = X * X, Y * Y, X * Y
        rx = A[0] * X2 + A[1] * XY + A[2] * Y2 + A[3] * X + A[4] * Y + A[5] - Xp
        ry = B[0] * X2 + B[1] * XY + B[2] * Y2 + B[3] * X + B[4] * Y + B[5] - Yp
        return [rx, ry]

    sol = least_squares(residual, x0, max_nfev=50)
    return sol.x


def map_mp_to_op_raw(pt_mp, A, B):
    """
    pt_mp: (row, col) in MP image
    returns: (row, col) in OP image
    NO montage origin involved
    """
    r, c = pt_mp
    X = c
    Y = r

    Xp, Yp = quadratic_transform(X, Y, A, B)

    return np.array([Yp, Xp])


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
    try:
        sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

        kp_m, desc_m = sift.detectAndCompute(moving, None)
        kp_f, desc_f = sift.detectAndCompute(fixed, None)

        if desc_m is None or desc_f is None:
            return None, False

        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn = bf.knnMatch(desc_m, desc_f, k=2)

        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        if len(good) < min_matches:
            return None, False

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

        if M is None or inliers.sum() < 8:
            return None, False

        T = np.eye(3)
        T[:2, :] = M
        return T, True

    except Exception as e:
        dbg(f"SURF error: {e}")
        return None, False


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


def try_phase_corr(moving, fixed):
    """
    MATLAB-equivalent phase correlation with resizing
    """
    h = min(moving.shape[0], fixed.shape[0])
    w = min(moving.shape[1], fixed.shape[1])

    if h < 32 or w < 32:
        return None, False

    moving_r = cv2.resize(moving, (w, h), interpolation=cv2.INTER_LINEAR)
    fixed_r = cv2.resize(fixed, (w, h), interpolation=cv2.INTER_LINEAR)

    shift, _, _ = phase_cross_correlation(fixed_r, moving_r)

    # Scale shift back to original resolution
    sy = fixed.shape[0] / h
    sx = fixed.shape[1] / w

    T = np.eye(3)
    T[:2, 2] = [shift[1] * sx, shift[0] * sy]

    return T, True


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
    T, ok = try_surf(mp_c, op_c)

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

    warped = warp(
        mp_full, tform.inverse, output_shape=op_full.shape, preserve_range=True
    ).astype(np.uint8)

    return warped, tform, crop_m, crop_f


# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(fixed, warped):
    return {
        "NMI": normalized_mutual_information(fixed, warped),
        "SSIM": structural_similarity(fixed, warped, data_range=255),
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


def visualize_combined(
    mp_full, op_full, tform_custom, crop_m, crop_f, transform_txt_i2k
):
    """
    mp_full, op_full: original grayscale images
    tform_custom, crop_m, crop_f: from your local registration
    transform_txt_i2k: i2k quadratic transform txt
    """

    # Parse i2k transforms
    transforms, montage_origin = parse_i2k_transform(transform_txt_i2k)
    mp_key = next(k for k in transforms if "MP" in k)
    A, B = transforms[mp_key]
    ox, oy = montage_origin

    mp_disp = cv2.cvtColor(mp_full, cv2.COLOR_GRAY2BGR)
    op_disp = cv2.cvtColor(op_full, cv2.COLOR_GRAY2BGR)

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        print(f"\nMP click: x={x}, y={y}")
        cv2.circle(mp_disp, (x, y), 4, (0, 0, 255), -1)  # clicked point = red

        # --- Custom registration mapping ---
        mapped_custom = map_mp_to_op_custom((y, x), tform_custom, crop_m, crop_f)
        r_c, c_c = mapped_custom
        xi_c, yi_c = int(round(c_c)), int(round(r_c))
        print(f"[CUSTOM] OP: x={xi_c}, y={yi_c}")
        if 0 <= xi_c < op_full.shape[1] and 0 <= yi_c < op_full.shape[0]:
            cv2.circle(op_disp, (xi_c, yi_c), 6, (0, 255, 0), -1)  # green = custom

        # --- i2k registration mapping ---
        Yp_i2k, Xp_i2k = map_mp_to_op_raw((y, x), A, B)
        xi_i2k, yi_i2k = int(round(Xp_i2k)), int(round(Yp_i2k))
        print(f"[i2k] OP: x={xi_i2k}, y={yi_i2k}")
        if 0 <= xi_i2k < op_full.shape[1] and 0 <= yi_i2k < op_full.shape[0]:
            cv2.circle(op_disp, (xi_i2k, yi_i2k), 6, (0, 0, 255), -1)  # red = i2k

    cv2.namedWindow("MP Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("OP Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MP Image", 900, 900)
    cv2.resizeWindow("OP Image", 900, 900)

    cv2.setMouseCallback("MP Image", on_click)

    while True:
        cv2.imshow("MP Image", mp_disp)
        cv2.imshow("OP Image", op_disp)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # --- Load MP/OP images ---
    dicom_mp = (
        r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\6812.dcm"
    )
    dicom_op = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\120101_203559_2451345_IR_OP_0.dcm"

    mp = force_2d(load_dicom_gray(dicom_mp))
    op = force_2d(load_dicom_gray(dicom_op))

    # --- Custom registration ---
    warped, tform_custom, crop_m, crop_f = register_custom(mp, op)

    # --- i2k registration ---
    mp_img_path, op_img_path, transform_txt = register_dicoms(dicom_mp, dicom_op)
    mp_i2k = imageio.imread(mp_img_path)
    op_i2k = imageio.imread(op_img_path)

    # --- Launch combined visualization ---
    visualize_combined(mp, op, tform_custom, crop_m, crop_f, transform_txt)

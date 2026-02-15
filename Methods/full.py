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
# STEP 1: CONVERT TO DICOM
# -----------------------------
def convert_to_dicom(file_path, format_type):
    dbg(f"Converting {file_path} ({format_type})")

    url = f"{CONVERSION_API}/{format_type}"
    headers = {"X-API-Key": CONVERSION_KEY}

    with open(file_path, "rb") as f:
        response = SESSION.post(
            url,
            headers=headers,
            files={
                "file": (os.path.basename(file_path), f, "application/octet-stream")
            },
            stream=True,
            timeout=(600, 3600),
            verify=False,
        )
    print("STATUS:", response.status_code)
    print("HEADERS:", response.headers)
    response.raise_for_status()

    if format_type == "e2e":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as zf:
            stream_to_file(response, zf.name)

        with zipfile.ZipFile(zf.name) as z:
            fundus = [n for n in z.namelist() if "345_IR_OP" in n]
            if not fundus:
                raise RuntimeError("Fundus DICOM not found in E2E zip")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as dcm:
                dcm.write(z.read(fundus[0]))
                return dcm.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as dcm:
        stream_to_file(response, dcm.name)
        return dcm.name


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
            print("Montage_origin : ", montage_origin)
        elif lines[i].startswith("IMAGE_NAME"):
            name = lines[i].split()[-1]
            if lines[i + 1] == "QUADRATIC":
                A = np.fromstring(lines[i + 2], sep=" ")
                B = np.fromstring(lines[i + 3], sep=" ")
                transforms[name] = (A, B)
                i += 3
        i += 1

    if montage_origin is None or not transforms:
        raise RuntimeError("Invalid transform file")

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


def map_mp_to_op(pt_mp, A, B, origin):
    """
    pt_mp: (row, col) in MP image
    returns: (row, col) in OP image
    """

    # 1. MP original → montage
    r, c = pt_mp
    Y = r + origin[0]  # row
    X = c + origin[1]  # col

    # 2. Quadratic transform (montage → montage)
    Xp, Yp = quadratic_transform(X, Y, A, B)

    # 3. Montage → OP original
    r_op = Yp - origin[0]
    c_op = Xp - origin[1]

    return np.array([r_op, c_op])


# -----------------------------
# VISUALIZATION
# -----------------------------
def visualize_debug(mp_img, op_img, transform_txt):
    # Parse transform
    transforms, montage_origin = parse_i2k_transform(transform_txt)
    mp_key = next(k for k in transforms if "MP" in k)
    A, B = transforms[mp_key]
    ox, oy = montage_origin

    h_mp, w_mp = mp_img.shape[:2]
    h_op, w_op = op_img.shape[:2]

    # Canvas size = just fit the largest image
    H = max(h_mp, h_op)
    W = max(w_mp, w_op)

    # Keep images top-left (no shifting)
    mp_disp = cv2.cvtColor(mp_img, cv2.COLOR_GRAY2BGR)
    op_disp = cv2.cvtColor(op_img, cv2.COLOR_GRAY2BGR)

    # Draw boundaries
    cv2.rectangle(mp_disp, (0, 0), (w_mp - 1, h_mp - 1), (255, 255, 0), 2)
    cv2.rectangle(op_disp, (0, 0), (w_op - 1, h_op - 1), (255, 255, 0), 2)

    # Mouse callback
    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        print(f"\nMP click: x={x}, y={y}")
        cv2.circle(mp_disp, (x, y), 4, (0, 255, 0), -1)

        # ----- WITH MONTAGE -----
        X = x + ox
        Y = y + oy
        Xp, Yp = quadratic_transform(X, Y, A, B)
        x_op_m = Xp - ox
        y_op_m = Yp - oy

        print(f"[WITH MONTAGE] OP: x={x_op_m:.2f}, y={y_op_m:.2f}")
        xi_m, yi_m = int(round(x_op_m)), int(round(y_op_m))
        if 0 <= xi_m < w_op and 0 <= yi_m < h_op:
            cv2.circle(op_disp, (xi_m, yi_m), 6, (0, 0, 255), -1)

        # ----- RAW EQUATION ONLY -----
        Yp_r, Xp_r = map_mp_to_op_raw((y, x), A, B)
        print(f"[RAW EQUATION] OP: x={Xp_r:.2f}, y={Yp_r:.2f}")
        xi_r, yi_r = int(round(Xp_r)), int(round(Yp_r))
        if 0 <= xi_r < w_op and 0 <= yi_r < h_op:
            cv2.circle(op_disp, (xi_r, yi_r), 6, (255, 0, 0), -1)

    cv2.namedWindow("MP (global frame)")
    cv2.namedWindow("OP (global frame)")
    cv2.setMouseCallback("MP (global frame)", on_click)

    while True:
        cv2.imshow("MP (global frame)", mp_disp)
        cv2.imshow("OP (global frame)", op_disp)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    e2e_path = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\OCT\LBS-008-CT03-120101-OCT-V5.e2e"
    tgz_path = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\MP\backup_maia-1121-patient1091_20240116\1091\6812.tgz"

    dicom_mp = convert_to_dicom(tgz_path, "tgz")
    dicom_op = convert_to_dicom(e2e_path, "e2e")

    mp_img_path, op_img_path, transform_txt = register_dicoms(dicom_mp, dicom_op)

    mp_img = imageio.imread(mp_img_path)
    op_img = imageio.imread(op_img_path)

    visualize_debug(mp_img, op_img, transform_txt)

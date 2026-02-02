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


# -----------------------------
# VISUALIZATION
# -----------------------------
def visualize_debug(mp_full, op_full, i2k_txt):

    transforms, _ = parse_i2k_transform(i2k_txt)
    mp_key = next(k for k in transforms if "MP" in k)
    A, B = transforms[mp_key]

    mp_disp = cv2.cvtColor(mp_full, cv2.COLOR_GRAY2BGR)
    op_disp = cv2.cvtColor(op_full, cv2.COLOR_GRAY2BGR)

    # -------- legend --------
    def draw_legend(img):
        cv2.putText(
            img, "Blue = Click", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )
        cv2.putText(
            img, "Red = i2k", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

    draw_legend(mp_disp)
    draw_legend(op_disp)

    # -------- mouse callback --------
    def on_click(event, x, y, flags, param):
        nonlocal mp_disp, op_disp

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if param == "MP":
            print(f"\nClicked MP: x={x}, y={y}")
            cv2.circle(mp_disp, (x, y), 5, (255, 0, 0), -1)

            # ---- i2k forward ----
            Yp, Xp = map_mp_to_op_raw((y, x), A, B)
            xi, yi = int(round(Xp)), int(round(Yp))
            print(f"[i2k] OP: x={xi}, y={yi}")
            if 0 <= xi < op_full.shape[1] and 0 <= yi < op_full.shape[0]:
                cv2.circle(op_disp, (xi, yi), 5, (0, 0, 255), -1)

        elif param == "OP":
            print(f"\nClicked OP: x={x}, y={y}")
            cv2.circle(op_disp, (x, y), 5, (255, 0, 0), -1)

            # ---- i2k inverse ----
            Xmp, Ymp = inverse_quadratic(x, y, A, B)
            xi, yi = int(round(Xmp)), int(round(Ymp))
            print(f"[i2k inverse] MP: x={xi}, y={yi}")
            if 0 <= xi < mp_full.shape[1] and 0 <= yi < mp_full.shape[0]:
                cv2.circle(mp_disp, (xi, yi), 6, (0, 0, 255), -1)

    # -------- windows --------
    cv2.namedWindow("MP", cv2.WINDOW_NORMAL)
    cv2.namedWindow("OP", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MP", 900, 900)
    cv2.resizeWindow("OP", 900, 900)

    cv2.setMouseCallback("MP", on_click, "MP")
    cv2.setMouseCallback("OP", on_click, "OP")

    while True:
        cv2.imshow("MP", mp_disp)
        cv2.imshow("OP", op_disp)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    dicom_mp = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\sample\MP\6812.dcm"
    dicom_op = r"F:\Burhan\OIRRC\Data\Multimodal_Data\CFP\PHOENIX_AU01001_CFP_SCR\PHOENIX_AU01001_19420101_20240409_1142_CX-1_Image_OD_1.2.392.200046.100.3.7.300387.3588.20240409114243.1.1.7.1.dcm"

    mp_img_path, op_img_path, transform_txt = register_dicoms(dicom_mp, dicom_op)

    mp_img = imageio.imread(mp_img_path)
    op_img = imageio.imread(op_img_path)

    visualize_debug(mp_img, op_img, transform_txt)

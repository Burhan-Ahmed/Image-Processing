import cv2
import numpy as np
import requests
import pydicom
import imageio

# -----------------------------
# CONFIG
# -----------------------------
REGISTRATION_API = "http://localhost:8000/in-house-registration/register"


# -----------------------------
# UTILITIES
# -----------------------------
def load_dicom_gray(path):
    ds = pydicom.dcmread(path, force=True)
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    img -= img.min()
    img /= max(img.max(), 1e-8)
    return (img * 255).astype(np.uint8)


# -----------------------------
# TRANSFORM MAPPING
# -----------------------------
def map_mp_to_op(pt_mp, T):
    """
    pt_mp: (row, col)
    T: 3x3 transform matrix
    returns: (row, col) in OP image
    """
    y, x = pt_mp
    vec = np.array([x, y, 1.0])
    xp, yp, w = T @ vec
    return np.array([yp / w, xp / w])  # row, col in OP


# -----------------------------
# INTERACTIVE VISUALIZATION
# -----------------------------
def visualize_with_matrix(mp_full, op_full, T):
    mp_disp = cv2.cvtColor(mp_full, cv2.COLOR_GRAY2BGR)
    op_disp = cv2.cvtColor(op_full, cv2.COLOR_GRAY2BGR)

    def on_click(event, x, y, flags, param):
        nonlocal mp_disp, op_disp
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if param == "MP":
            print(f"\nClicked MP: x={x}, y={y}")
            cv2.circle(mp_disp, (x, y), 5, (255, 0, 0), -1)
            yp, xp = map_mp_to_op((y, x), T)
            xi, yi = int(round(xp)), int(round(yp))
            print(f"Mapped OP: x={xi}, y={yi}")
            if 0 <= xi < op_full.shape[1] and 0 <= yi < op_full.shape[0]:
                cv2.circle(op_disp, (xi, yi), 5, (0, 255, 0), -1)
        elif param == "OP":
            print(f"\nClicked OP: x={x}, y={y}")
            cv2.circle(op_disp, (x, y), 5, (255, 0, 0), -1)
            # Inverse mapping using np.linalg.inv
            T_inv = np.linalg.inv(T)
            yp, xp = map_mp_to_op((y, x), T_inv)
            xi, yi = int(round(xp)), int(round(yp))
            print(f"Mapped MP: x={xi}, y={yi}")
            if 0 <= xi < mp_full.shape[1] and 0 <= yi < mp_full.shape[0]:
                cv2.circle(mp_disp, (xi, yi), 5, (0, 255, 0), -1)

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
    dicom_op = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\sample\MP\6812.dcm"
    dicom_mp = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\sample\OCT\120101_203559_2451345_IR_OP_0.dcm"

    # --- Load files and upload to API ---
    with open(dicom_mp, "rb") as f_mp, open(dicom_op, "rb") as f_op:
        files = {"mp_file": f_mp, "op_file": f_op}
        resp = requests.post(REGISTRATION_API, files=files, timeout=300)
        resp.raise_for_status()
        T = np.array(resp.json()["transform_matrix"])

    # --- Load images locally for visualization ---
    mp_img = load_dicom_gray(dicom_mp)
    op_img = load_dicom_gray(dicom_op)

    # --- Visualize and map points ---
    visualize_with_matrix(mp_img, op_img, T)

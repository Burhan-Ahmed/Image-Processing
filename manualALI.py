import cv2
import numpy as np
import pydicom
import os
import pandas as pd
import json
from scipy.interpolate import RBFInterpolator

# =========================
# SETTINGS
# =========================
CANVAS_WIDTH = 1804
CANVAS_HEIGHT = 1803


# =========================
# LOAD IMAGE FUNCTION
# =========================
def load_image(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".dcm":
        ds = pydicom.dcmread(path, force=True)

        if not hasattr(ds, "PixelData"):
            raise ValueError("DICOM has no PixelData")

        try:
            img = ds.pixel_array
        except RuntimeError as e:
            raise RuntimeError(
                "\nDICOM decompression failed.\n"
                "Install required decoders:\n"
                "  pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg\n"
                "OR\n"
                "  pip install gdcm\n"
                f"\nOriginal error:\n{e}"
            )

        print(f"Original shape: {img.shape}, dtype: {img.dtype}")

        # Handle multiframe properly
        if img.ndim == 3 and img.shape[0] <= 10 and img.shape[1] > 100:
            print(
                f"Detected multi-frame DICOM ({img.shape[0]} frames). Using first frame."
            )
            img = img[0]
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        # Normalize
        img = img.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

        # Ensure 3-channel
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Fix color for RGB DICOMs
        if img.ndim == 3 and img.shape[2] == 3:
            if getattr(ds, "PhotometricInterpretation", "") == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        print(f"Loaded image shape: {img.shape}")
        return img

    else:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load image: {path}")
        return img


# =========================
# TRANSFORMATION FUNCTIONS
# =========================
def build_A_quadratic(pts):
    return np.array([[x**2, y**2, x * y, x, y, 1] for x, y in pts])


def apply_tps_transform(img, src_pts, dst_pts, output_shape):
    h_out, w_out = output_shape
    h_in, w_in = img.shape[:2]

    rbf_x = RBFInterpolator(dst_pts, src_pts[:, 0], kernel="thin_plate_spline")
    rbf_y = RBFInterpolator(dst_pts, src_pts[:, 1], kernel="thin_plate_spline")

    y_out, x_out = np.mgrid[0:h_out, 0:w_out]
    out_coords = np.column_stack([x_out.ravel(), y_out.ravel()])

    src_x = rbf_x(out_coords).reshape(h_out, w_out)
    src_y = rbf_y(out_coords).reshape(h_out, w_out)

    valid_mask = (src_x >= 0) & (src_x < w_in) & (src_y >= 0) & (src_y < h_in)

    map_x = np.clip(src_x, 0, w_in - 1).astype(np.float32)
    map_y = np.clip(src_y, 0, h_in - 1).astype(np.float32)

    output = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    output[~valid_mask] = (0, 0, 0)
    return output


# =========================
# INPUT PATHS
# =========================
img_faf_path = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\FAF\LBS-008-CT03-120101-IR-V5_converted\120101_203559_2451353_IR_OP_0.dcm"
img_mp_path = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\OCT\LBS-008-CT03-120101-OCT-V5_converted\120101_203559_2451343_fundus_0.dcm"

img_faf_original = load_image(img_faf_path)
img_mp_original = load_image(img_mp_path)

# =========================
# OUTPUT FOLDER
# =========================
base_dir = input("Enter base directory path: ").strip()
if not base_dir:
    raise ValueError("Base directory cannot be empty.")

output_folder = os.path.join(base_dir, "output")
os.makedirs(output_folder, exist_ok=True)

# =========================
# SAVE ORIGINAL IMAGES AT NATIVE RESOLUTION
# =========================
cv2.imwrite(os.path.join(output_folder, "FAF_IR_image.png"), img_faf_original)
cv2.imwrite(os.path.join(output_folder, "MP_IR_image.png"), img_mp_original)

faf_h, faf_w = img_faf_original.shape[:2]
mp_h, mp_w = img_mp_original.shape[:2]

print(f"\n[INFO] Original image dimensions:")
print(f"  FAF: {faf_w} × {faf_h}")
print(f"  MP:  {mp_w} × {mp_h}")

# =========================
# RESIZE FOR LANDMARK SELECTION
# =========================
img_faf_canvas = cv2.resize(img_faf_original, (CANVAS_WIDTH, CANVAS_HEIGHT))
img_mp_canvas = cv2.resize(img_mp_original, (CANVAS_WIDTH, CANVAS_HEIGHT))

# =========================
# LANDMARK SELECTION
# =========================
canvas = np.hstack([img_faf_canvas, img_mp_canvas])
display = canvas.copy()

points_faf, points_mp = [], []
point_id = 1
waiting_for = "FAF"


def redraw_canvas():
    global display
    display = canvas.copy()

    for pid, x, y in points_faf:
        cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            display,
            str(pid),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    for pid, x, y in points_mp:
        cv2.circle(display, (x + CANVAS_WIDTH, y), 6, (0, 0, 255), -1)
        cv2.putText(
            display,
            str(pid),
            (x + CANVAS_WIDTH + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )


def mouse_callback(event, x, y, flags, param):
    global point_id, waiting_for

    if event == cv2.EVENT_LBUTTONDOWN:
        if waiting_for == "FAF" and x < CANVAS_WIDTH:
            points_faf.append((point_id, x, y))
            waiting_for = "MP"
            redraw_canvas()
            print(f"✓ Point {point_id}: FAF ({x}, {y})")
        elif waiting_for == "MP" and x >= CANVAS_WIDTH:
            mp_x = x - CANVAS_WIDTH
            points_mp.append((point_id, mp_x, y))
            point_id += 1
            waiting_for = "FAF"
            redraw_canvas()
            print(f"✓ Point {point_id-1}: MP ({mp_x}, {y})")

    if event == cv2.EVENT_RBUTTONDOWN:
        if waiting_for == "MP" and points_faf:
            removed = points_faf.pop()
            waiting_for = "FAF"
            redraw_canvas()
            print(f"✗ Removed point {removed[0]}")
        elif waiting_for == "FAF" and points_mp:
            removed = points_mp.pop()
            point_id -= 1
            waiting_for = "MP"
            redraw_canvas()
            print(f"✗ Removed point {removed[0]}")


print("\n[INFO] Landmark selection mode:")
print(
    "  • Click on FAF (left image) first, then corresponding point on MP (right image)"
)
print("  • Right-click to undo last point")
print("  • Press 'q' when done")

cv2.namedWindow("Landmark Selection: FAF | MP", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Landmark Selection: FAF | MP", mouse_callback)

while True:
    cv2.imshow("Landmark Selection: FAF | MP", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

if len(points_faf) != len(points_mp) or len(points_faf) < 3:
    raise ValueError("Need at least 3 matching landmark pairs")

pts_faf = np.array([[x, y] for _, x, y in points_faf], dtype=np.float32)
pts_mp = np.array([[x, y] for _, x, y in points_mp], dtype=np.float32)

print(f"\n[INFO] Collected {len(points_faf)} landmark pairs")

# =========================
# SCALE LANDMARKS TO ORIGINAL COORDINATES
# =========================
scale_x_faf = faf_w / CANVAS_WIDTH
scale_y_faf = faf_h / CANVAS_HEIGHT
scale_x_mp = mp_w / CANVAS_WIDTH
scale_y_mp = mp_h / CANVAS_HEIGHT

pts_faf_orig = pts_faf * [scale_x_faf, scale_y_faf]
pts_mp_orig = pts_mp * [scale_x_mp, scale_y_mp]

print(f"[INFO] Scale factors:")
print(f"  FAF: {scale_x_faf:.4f} × {scale_y_faf:.4f}")
print(f"  MP:  {scale_x_mp:.4f} × {scale_y_mp:.4f}")

# =========================
# COMPUTE MP → FAF TRANSFORMATION (ORIGINAL COORDINATES)
# =========================
# This is the key fix: compute transformation in original image space
A_mp_orig = build_A_quadratic(pts_mp_orig)
cx_mp_orig, _, _, _ = np.linalg.lstsq(A_mp_orig, pts_faf_orig[:, 0], rcond=None)
cy_mp_orig, _, _, _ = np.linalg.lstsq(A_mp_orig, pts_faf_orig[:, 1], rcond=None)

print("\n[INFO] Quadratic coefficients (MP → FAF) in original coordinates:")
print(f"  X: {cx_mp_orig}")
print(f"  Y: {cy_mp_orig}")

# =========================
# COMPUTE TRANSFORMATION IN CANVAS SPACE (FOR VISUALIZATION)
# =========================
A_mp_canvas = build_A_quadratic(pts_mp)
cx_mp_canvas, _, _, _ = np.linalg.lstsq(A_mp_canvas, pts_faf[:, 0], rcond=None)
cy_mp_canvas, _, _, _ = np.linalg.lstsq(A_mp_canvas, pts_faf[:, 1], rcond=None)

# Transform MP landmarks to FAF space (canvas coordinates)
transformed_mp_pts = np.array(
    [
        np.array([x**2, y**2, x * y, x, y, 1])
        @ np.vstack([cx_mp_canvas, cy_mp_canvas]).T
        for x, y in pts_mp
    ]
)

# =========================
# LANDMARK RMSE (CANVAS SPACE)
# =========================
landmark_rmse = np.sqrt(np.mean(np.sum((pts_faf - transformed_mp_pts) ** 2, axis=1)))
print(f"\n[INFO] Landmark RMSE (canvas space): {landmark_rmse:.4f} pixels")

# =========================
# APPLY TRANSFORMATION FOR VISUALIZATION
# =========================
# Keep FAF unchanged (it's the anchor)
registered_faf = img_faf_canvas.copy()

# Transform MP to align with FAF
registered_mp = apply_tps_transform(
    img_mp_canvas, pts_mp, pts_faf, (CANVAS_HEIGHT, CANVAS_WIDTH)
)

print("[INFO] Applied TPS transformation for visualization")

# =========================
# PIXEL-WISE RMSE
# =========================
faf_gray = cv2.cvtColor(registered_faf, cv2.COLOR_BGR2GRAY)
mp_gray = cv2.cvtColor(registered_mp, cv2.COLOR_BGR2GRAY)
valid_mask = (faf_gray > 0) & (mp_gray > 0)

pixel_rmse = (
    np.sqrt(
        np.mean(
            (
                faf_gray[valid_mask].astype(np.float32)
                - mp_gray[valid_mask].astype(np.float32)
            )
            ** 2
        )
    )
    if np.any(valid_mask)
    else np.nan
)
print(f"[INFO] Pixel-wise RMSE: {pixel_rmse:.4f} intensity units")

# =========================
# SAVE VISUALIZATION RESULTS
# =========================
cv2.imwrite(os.path.join(output_folder, "FAF_IR_mapped.png"), registered_faf)
cv2.imwrite(os.path.join(output_folder, "MP_IR_mapped.png"), registered_mp)

# =========================
# SAVE LANDMARKS
# =========================
pd.DataFrame(
    {
        "point_id": [i for i, _, _ in points_faf],
        "FAF_x_canvas": [x for _, x, _ in points_faf],
        "FAF_y_canvas": [y for _, _, y in points_faf],
        "MP_x_canvas": [x for _, x, _ in points_mp],
        "MP_y_canvas": [y for _, _, y in points_mp],
        "FAF_x_orig": pts_faf_orig[:, 0],
        "FAF_y_orig": pts_faf_orig[:, 1],
        "MP_x_orig": pts_mp_orig[:, 0],
        "MP_y_orig": pts_mp_orig[:, 1],
    }
).to_csv(os.path.join(output_folder, "landmarks.csv"), index=False)

# =========================
# SAVE COEFFICIENTS
# =========================
with open(os.path.join(output_folder, "quadratic_coefficients.json"), "w") as f:
    json.dump(
        {
            "MP_to_FAF_original_coords": {
                "x_coeff": cx_mp_orig.tolist(),
                "y_coeff": cy_mp_orig.tolist(),
            },
            "MP_to_FAF_canvas_coords": {
                "x_coeff": cx_mp_canvas.tolist(),
                "y_coeff": cy_mp_canvas.tolist(),
            },
            "original_dimensions": {"FAF": [faf_w, faf_h], "MP": [mp_w, mp_h]},
            "canvas_size": [CANVAS_WIDTH, CANVAS_HEIGHT],
            "scale_factors": {
                "FAF": [scale_x_faf, scale_y_faf],
                "MP": [scale_x_mp, scale_y_mp],
            },
            "note": "FAF is anchor (unchanged). MP transformed to FAF space.",
        },
        f,
        indent=4,
    )

# =========================
# SAVE RMSE
# =========================
with open(os.path.join(output_folder, "rmse.txt"), "w") as f:
    f.write(f"Landmark RMSE (canvas space): {landmark_rmse:.4f} pixels\n")
    f.write(f"Pixel-wise RMSE (grayscale): {pixel_rmse:.4f}\n")
    f.write("\nRegistration method:\n")
    f.write("  • FAF is anchor (kept at original coordinates)\n")
    f.write("  • MP transformed to align with FAF\n")
    f.write("  • Transformation computed in ORIGINAL image coordinates\n")
    f.write("  • Compatible with grid mapping code\n")

# =========================
# TRANSFORMATION FILE (i2k Retina format)
# =========================
faf_name = "FAF_IR_image.png"
mp_name = "MP_IR_image.png"

with open(os.path.join(output_folder, "transformation.txt"), "w") as f:
    f.write("NUMBER_OF_IMAGES 2\n")
    f.write("MONTAGE_ORIGIN 0 0\n")
    f.write(f"ANCHOR_IMAGE_NAME {faf_name}\n")

    # MP transformation (using ORIGINAL coordinate coefficients)
    f.write(f"\nIMAGE_NAME {mp_name}\n")
    f.write("QUADRATIC\n")
    f.write(" ".join([f"{v:.8e}" for v in cx_mp_orig]) + "\n")
    f.write(" ".join([f"{v:.8e}" for v in cy_mp_orig]) + "\n")

    # FAF is anchor (identity transform)
    f.write(f"\nIMAGE_NAME {faf_name}\n")
    f.write("QUADRATIC\n")
    f.write("0 0 0 1 0 0\n")
    f.write("0 0 0 0 1 0\n")

print("[INFO] Saved transformation.txt in i2k Retina format")

# =========================
# SUMMARY
# =========================
print("\n" + "=" * 70)
print("✓ REGISTRATION COMPLETE!")
print("=" * 70)
print(f"Output directory: {output_folder}")
print(f"\nLandmarks: {len(points_faf)} pairs")
print(f"Landmark RMSE (canvas): {landmark_rmse:.4f} pixels")
print(f"Pixel-wise RMSE: {pixel_rmse:.4f}")
print(f"\nCanvas size: {CANVAS_WIDTH} × {CANVAS_HEIGHT}")
print(f"\nOriginal image dimensions:")
print(f"  • FAF: {faf_w} × {faf_h}")
print(f"  • MP:  {mp_w} × {mp_h}")
print(f"\nRegistration method:")
print(f"  • FAF is ANCHOR (remains at original coordinates)")
print(f"  • MP transformed to align with FAF")
print(f"  • Transformation computed in ORIGINAL image space")
print(f"  • Grid mapping code will now work correctly")
print(f"\nOutput files:")
print(f"  • FAF_IR_image.png (original at {faf_w}×{faf_h})")
print(f"  • MP_IR_image.png (original at {mp_w}×{mp_h})")
print(f"  • FAF_IR_mapped.png (visualization on canvas)")
print(f"  • MP_IR_mapped.png (visualization on canvas)")
print(f"  • transformation.txt (i2k Retina format - FOR GRID MAPPING)")
print(f"  • landmarks.csv (both canvas and original coordinates)")
print(f"  • quadratic_coefficients.json (detailed coefficients)")
print(f"  • rmse.txt (quality metrics)")
print("=" * 70)
print("\n[IMPORTANT] Use 'transformation.txt' with your grid mapping code!")
print("=" * 70 + "\n")

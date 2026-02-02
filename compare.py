import time
import numpy as np
import imageio.v2 as imageio
import cv2

# -----------------------------
# IMPORT CUSTOM PIPELINE
# -----------------------------
from DR_hikmat import register_custom, compute_metrics

# -----------------------------
# IMPORT I2K CLIENT
# -----------------------------
from direct import (
    register_dicoms,
    parse_i2k_transform,
    quadratic_transform,
)


# -----------------------------
# I2K WARP (MP → OP)
# -----------------------------
def warp_mp_using_i2k(mp_img, op_img, transform_txt):
    transforms, montage_origin = parse_i2k_transform(transform_txt)

    mp_key = next(k for k in transforms if "MP" in k)
    A, B = transforms[mp_key]

    h_op, w_op = op_img.shape[:2]

    yy, xx = np.meshgrid(np.arange(h_op), np.arange(w_op), indexing="ij")

    # Inverse mapping (OP → MP)
    map_y = np.zeros_like(yy, dtype=np.float32)
    map_x = np.zeros_like(xx, dtype=np.float32)

    for r in range(h_op):
        for c in range(w_op):
            Xp = c
            Yp = r
            X, Y = quadratic_transform(Xp, Yp, A, B)
            map_x[r, c] = X
            map_y[r, c] = Y

    warped = cv2.remap(
        mp_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return warped


# -----------------------------
# RUN I2K REGISTRATION
# -----------------------------
def run_i2k(mp_dicom, op_dicom):
    t0 = time.time()

    mp_img_path, op_img_path, transform_txt = register_dicoms(mp_dicom, op_dicom)

    mp_img = imageio.imread(mp_img_path)
    op_img = imageio.imread(op_img_path)

    warped = warp_mp_using_i2k(mp_img, op_img, transform_txt)

    elapsed = time.time() - t0

    return {
        "success": True,
        "mp": mp_img,
        "op": op_img,
        "warped": warped,
        "time_sec": elapsed,
    }


# -----------------------------
# RUN CUSTOM REGISTRATION
# -----------------------------
def run_custom(mp_img, op_img):
    t0 = time.time()

    res = register_custom(mp_img, op_img)

    elapsed = time.time() - t0

    return {
        "success": True,
        "mp": res["mp_original"],
        "op": res["op_original"],
        "warped": res["mp_mapped"],
        "num_features": res["num_features"],
        "time_sec": elapsed,
    }


# -----------------------------
# COMPARE BOTH METHODS
# -----------------------------
def compare_results(i2k_res, custom_res, num_features_custom=None):
    out = {}

    # I2K: no features returned, keep same
    out["I2K"] = {
        "time_sec": i2k_res["time_sec"],
        **compute_metrics(i2k_res["op"], i2k_res["warped"]),
    }

    # CUSTOM: pass number of matched features
    out["CUSTOM"] = {
        "time_sec": custom_res["time_sec"],
        **compute_metrics(
            custom_res["op"],
            custom_res["warped"],
            num_features=num_features_custom,
        ),
    }

    return out


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    dicom_mp = (
        r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\6812.dcm"
    )
    dicom_op = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\120101_203559_2451345_IR_OP_0.dcm"

    print("\n--- Running I2K ---")
    i2k_res = run_i2k(dicom_mp, dicom_op)

    print("\n--- Running CUSTOM ---")
    custom_res = run_custom(i2k_res["mp"], i2k_res["op"])

    print("\n--- COMPARISON ---")
    results = compare_results(
        i2k_res, custom_res, num_features_custom=custom_res.get("num_features", 0)
    )

    for k, v in results.items():
        print(f"\n{k}")
        for kk, vv in v.items():
            if isinstance(vv, float):
                print(f"  {kk}: {vv:.4f}")
            else:
                print(f"  {kk}: {vv}")

        for k, v in results.items():
            print(f"\n{k}")
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")

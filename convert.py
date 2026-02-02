import pydicom
from pydicom.uid import generate_uid


def save_first_frame_dicom(input_path, output_path):
    """
    Extract the first frame of a multi-frame DICOM and save as a new single-frame DICOM.
    Metadata is preserved.
    """
    ds = pydicom.dcmread(input_path, force=True)
    pixel_array = ds.pixel_array

    # Take only the first frame if multi-frame
    if pixel_array.ndim == 3:  # (frames, rows, cols)
        first_frame = pixel_array[0]
    else:
        first_frame = pixel_array

    # Update PixelData and Rows/Columns
    ds.PixelData = first_frame.tobytes()
    ds.Rows, ds.Columns = first_frame.shape

    # Remove multi-frame tags
    if hasattr(ds, "NumberOfFrames"):
        del ds.NumberOfFrames

    # Ensure unique SOP Instance UID for new DICOM
    ds.SOPInstanceUID = generate_uid()

    # Save new DICOM
    ds.save_as(output_path)
    print(f"Saved single-frame DICOM to: {output_path}")


# Example usage
input_dcm = (
    r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\sample\MP\6812.dcm"
)
output_dcm = r"F:\Burhan\OIRRC\Data\CT03-Patient-Data-Dr.Najam\120101\Visit 5\sample\MP\first_frame.dcm"
save_first_frame_dicom(input_dcm, output_dcm)

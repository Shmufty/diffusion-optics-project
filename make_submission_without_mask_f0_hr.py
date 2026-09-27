"""Build submission_without_mask_f0_hr/'s tissue file: submission/'s
full-resolution (351x351, x_stp=0.1um) two-RBC tissue .mat with the
mask_f0_hr variable removed. Nothing is resampled or recomputed: every
remaining variable is copied bit-for-bit.

mask_f0_hr (the fine-z-grid, fluctuation-only mask -- ~281MB of the
~559MB file) is read by functions.py's load_background but never used in
any computation (only mask_f_hr, mask_f and the grid scalars are), so
dropping it should leave every result bit-identical to submission/'s.
That's what this variant tests: a ~277MB alternative to submission_lite/
(XY resampling) that keeps full resolution. The saved file gets a new,
self-describing name (tissue_background_35x35x15_two_rbcs_without_
mask_f0_hr.mat); submission_without_mask_f0_hr/functions.py points
MAT_PATH at it and no longer reads mask_f0_hr in load_background.

Run from anywhere: python make_submission_without_mask_f0_hr.py
"""
import os
import time

import h5py

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_MAT = os.path.join(ROOT, "submission", "capillary_system", "tissue_background_35x35x15_two_rbcs.mat")
DST_MAT = os.path.join(
    ROOT, "submission_without_mask_f0_hr", "capillary_system",
    "tissue_background_35x35x15_two_rbcs_without_mask_f0_hr.mat")

DROPPED = "mask_f0_hr"


def write_mat_header(path):
    """MATLAB only recognizes a -v7.3 file by the text header in its HDF5
    userblock -- same byte layout MATLAB itself writes (116-byte text,
    8-byte subsys offset, version 0x0200, 'IM' endian tag), fresh date."""
    text = (f"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: "
            f"{time.strftime('%a %b %d %H:%M:%S %Y')} HDF5 schema 1.00 .")
    header = text.ljust(116).encode("ascii") + b"\x00" * 8 + b"\x00\x02" + b"IM"
    with open(path, "r+b") as f:
        f.write(header.ljust(512, b"\x00"))


def main():
    os.makedirs(os.path.dirname(DST_MAT), exist_ok=True)
    with h5py.File(SRC_MAT, "r") as src, h5py.File(DST_MAT, "w", userblock_size=512) as dst:
        for name, ds in src.items():
            if name != DROPPED:
                src.copy(ds, dst, name=name)
    write_mat_header(DST_MAT)

    print(f"Saved {DST_MAT} without {DROPPED} ({os.path.getsize(DST_MAT) / 1e6:.1f} MB, "
          f"was {os.path.getsize(SRC_MAT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

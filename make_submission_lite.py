"""Build submission_lite/'s tissue file: submission/'s two-RBC background
tissue (capillary_system/tissue_background_35x35x15_two_rbcs.mat),
resampled from 351x351 to 281x281 in XY (x_stp 0.1um -> 0.125um), so the
.mat drops from ~559MB to ~359MB. Everything else about the tissue is
unchanged: same 35x35um field of view, same random sphere realization,
same 150 fine z-layers at 0.1um (75 per aberration plane), and the same
filename -- so submission_lite/'s code (a verbatim copy of submission/'s)
runs on it unmodified.

The resampling is band-limited (periodic sinc, i.e. Dirichlet-kernel)
interpolation, which is EXACT here, not an approximation:
tissue_3d_generator.m low-pass filters every fine layer in XY (its OTF
mask keeps only |f| < 0.5*Nyquist = 2.5 cycles/um at x_stp=0.1um), using
FFTs over the full N=351-sample grid -- so each layer is exactly a
trigonometric polynomial of period N*x_stp, and evaluating that
polynomial at the new grid points gives exact samples of the same
continuous tissue. The new grid's Nyquist (1/(2*0.125um) = 4 cycles/um)
is well above 2.5, so those samples lose nothing. The coarse (M=2)
planes mask_f/mask_f0 are linear z-averages of the fine layers, so the
same holds for them.

Only the four tissue volumes (mask_f_hr, mask_f0_hr, mask_f, mask_f0)
and x_stp change; every other variable is copied verbatim. The output
keeps MATLAB's -v7.3 layout (HDF5 + 512-byte MAT header, MATLAB_class
attributes, the source's own gzip level/chunking), so MATLAB's load()
reads it just like the original.

Run from anywhere: python make_submission_lite.py
"""
import os
import time

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_MAT = os.path.join(ROOT, "submission", "capillary_system", "tissue_background_35x35x15_two_rbcs.mat")
DST_MAT = os.path.join(ROOT, "submission_lite", "capillary_system", "tissue_background_35x35x15_two_rbcs.mat")

# 281 samples over the same 35um -> x_stp = 35/280 = 0.125um exactly. Odd,
# so the point source at pixel N//2 still sits exactly on x=0.
N_LITE = 281
RESAMPLED = ("mask_f_hr", "mask_f0_hr", "mask_f", "mask_f0")


def dirichlet_matrix(x_new, x_old, period):
    """(len(x_new), len(x_old)) matrix P such that P @ v evaluates, at
    x_new, the unique trigonometric polynomial of the given period that
    passes through samples v taken at x_old. len(x_old) must be odd (no
    ambiguous Nyquist term) -- 351 here."""
    n = x_old.shape[0]
    d = x_new[:, None] - x_old[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.sin(np.pi * n * d / period) / (n * np.sin(np.pi * d / period))
    P[np.isclose(d, 0.0, atol=1e-9)] = 1.0
    return P


def resample_xy(raw, P):
    """raw is h5py's view of a MATLAB (N, N, Nz) complex array: axes
    reversed to (Nz, N, N), real/imag as compound fields. Both XY axes
    share one grid, so both get the same P: out[z] = P @ raw[z] @ P.T."""
    out = np.empty((raw.shape[0], P.shape[0], P.shape[0]), dtype=raw.dtype)
    for part in ("real", "imag"):
        out[part] = P @ raw[part] @ P.T
    return out


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
        x_max = float(np.asarray(src["x_max"]).squeeze())
        x_stp = float(np.asarray(src["x_stp"]).squeeze())
        n_src = src["mask_f_hr"].shape[-1]
        x_stp_lite = 2 * x_max / (N_LITE - 1)
        # Period = n_src*x_stp (35.1um), NOT 2*x_max (35um): that's the
        # period the generator's own FFT filtering imposed on the data.
        P = dirichlet_matrix(
            np.linspace(-x_max, x_max, N_LITE), np.linspace(-x_max, x_max, n_src), n_src * x_stp)

        for name, ds in src.items():
            if name in RESAMPLED:
                data = resample_xy(ds[:], P)
                new = dst.create_dataset(
                    name, data=data,
                    chunks=tuple(min(c, s) for c, s in zip(ds.chunks, data.shape)),
                    compression=ds.compression, compression_opts=ds.compression_opts, shuffle=ds.shuffle)
            elif name == "x_stp":
                new = dst.create_dataset(name, data=np.full(ds.shape, x_stp_lite))
            else:
                src.copy(ds, dst, name=name)
                continue
            for key, val in ds.attrs.items():
                new.attrs[key] = val
    write_mat_header(DST_MAT)

    print(f"Resampled {n_src}x{n_src} (x_stp={x_stp:g}um) -> {N_LITE}x{N_LITE} (x_stp={x_stp_lite:g}um)")
    print(f"Saved {DST_MAT} ({os.path.getsize(DST_MAT) / 1e6:.1f} MB, "
          f"was {os.path.getsize(SRC_MAT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

===============================================================================
 Diffusion Optics -- Project Submission
 Robustness of a fixed (t=0-calibrated) two-plane wavefront correction
 under red-blood-cell flow
===============================================================================


HOW TO RUN
-------------------------------------------------------------------------------
1. Build a Python environment from requirements.txt, which is the complete
   specification of what the environment must consist of -- nothing beyond
   those four packages is needed:

       python -m venv venv
       venv\Scripts\activate            (Windows)
       source venv/bin/activate         (Linux / macOS)
       pip install -r requirements.txt

2. Run the single entry point:

       python main.py

   That is all that is required. Running main.py performs the entire
   simulation and produces every result in this submission: all figures are
   written to the results/ folder, the animation is written beside main.py,
   and the quantitative robustness table is printed to the console. No other
   script needs to be run, and no arguments or configuration are needed.
   Expect a few minutes of runtime; the figures are also shown on screen at
   the end.


FILES
-------------------------------------------------------------------------------
main.py
    The single entry point: runs the full propagate / aberrate / correct /
    backpropagate pipeline across all 17 timesteps and generates every output
    listed below.

functions.py
    All supporting code main.py imports -- angular-spectrum propagation,
    capillary and red-blood-cell geometry and timing, aberration-plane
    reconstruction, and the robustness analysis.

requirements.txt
    The exact package versions the environment must consist of: numpy, h5py,
    matplotlib and Pillow.

README.txt
    This file.

Submission_Omer.pdf
    The written report accompanying the code.

capillary_system/tissue_background_35x35x15_two_rbcs.mat
    The pre-generated 35x35x15um tissue volume (background scatterers plus
    capillary geometry) that main.py loads, included so this folder runs
    as-is without having to regenerate it in MATLAB.

rbc_flow_speckle_animation.gif
    Animated side-by-side comparison of the corrected and uncorrected speckle
    fields across the 17 timesteps, written here by main.py.


RESULTS -- all written by main.py into results/
-------------------------------------------------------------------------------
input_field.png
    The ideal point-source input field, shown full-frame and cropped around
    its centre.

aberration_plane_A.png
    Phase of aberration plane A (tissue entrance) at every one of the 17
    timesteps.

aberration_plane_B.png
    Phase of aberration plane B (tissue exit face) at every one of the 17
    timesteps.

aberration_plane_A_4_timesteps.png
    Phase of plane A at four evenly spaced timesteps spanning t=0 to t=4.0s,
    as a readable summary of the full 17-panel figure.

rbc_flow_correction_u_out.png
    Output field amplitude at every timestep with the fixed t=0-calibrated
    correction applied, all panels on one shared colour scale.

rbc_flow_correction_u_out_cropped50.png
    The same data cropped to a 50x50 pixel window around the focus, where the
    spot is actually visible against the 351x351 grid.

rbc_flow_no_correction_u_out.png
    The uncorrected baseline -- light through the tissue backpropagated with
    no correction planes at all -- on the same colour scale as the corrected
    figure, so the two are directly comparable.

rbc_flow_correction_u_out_propagated2eps.png
    The corrected field carried a further +2*epsilon past the refocus plane,
    which is the extended speckle pattern the g2 statistics are computed on.

rbc_flow_no_correction_u_out_propagated2eps.png
    The same further propagation applied to the uncorrected path.

speckle_correlation_statistics_propagated2eps.png
    The intensity autocorrelation g2(tau) of both paths overlaid, for a direct
    comparison of how fast each decorrelates.

correction_robustness_vs_drift.png
    The main robustness result: correction efficiency (Strehl) against elapsed
    time, and then against how far the sample has drifted from its calibration
    state, so the estimate is not tied to this run's particular flow timing.

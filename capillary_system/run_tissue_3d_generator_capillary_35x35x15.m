% run_tissue_3d_generator_capillary_35x35x15: Generates the M=2 base
% tissue (t=0, no capillary/cell yet) for sim step 2, at 35x35x15um.
% The capillary + moving red blood cell are carved in separately, per
% timestep, in Python (step_two.py) -- this script only produces the
% static random-sphere background, at BOTH fine (pre-downsampling) and
% coarse (M=2 plane) resolution, since the capillary/cell carving needs
% to happen at fine resolution before being re-averaged into 2 planes
% (see step_two.py for why: coarse averaging must happen with the hard,
% unsmoothed capillary edges already in place, but tissue_3d_generator's
% own OTF/NA smoothing filter has already run by the time mask_f_hr is
% captured -- so carving post-hoc, after this script, is the only way to
% get genuinely unsmoothed capillary walls as the guidelines specify).
%
% Same conventions as run_tissue_3d_generator_2planes_50x50x10.m:
% forward propagation coordinates (z=0 at entrance), GPU auto-selection,
% and is_cyclic=1 for memory (this tissue's non-cyclic working grid would
% be ~701x701x300 =~1.47e8 voxels -- too close to the 50x50x10 non-cyclic
% case that already OOM'd even on a fully-free 24GB GPU; cyclic here is
% ~351x351x300 =~3.7e7 voxels, smaller than the 50x50x10 cyclic case that
% already worked comfortably).

%% === Pick the GPU with the most free memory ===
n_gpus = gpuDeviceCount();
free_mem = zeros(1, n_gpus);
for gi = 1:n_gpus
    d = gpuDevice(gi);
    free_mem(gi) = d.AvailableMemory;
end
[~, best_gpu] = max(free_mem);
gpuDevice(best_gpu);
fprintf('Using GPU %d (%.1f GB free)\n', best_gpu, free_mem(best_gpu)/1e9);

%% === Set parameters ===
x_max = 17.5;      % Maximum extent in x/y [um] -> lateral width 35
x_stp = 0.1;         % Sampling step size in x/y [um] (unchanged from prior tissues)
z_max = 7.5;         % Tissue half-width in z [um] -> full depth 15
M = 2;                % Number of aberration planes
Z_tissue = 2*z_max;      % Full tissue width, entrance (z=0) to exit (z=Z_tissue)
Delta = Z_tissue/M;      % Spacing between planes == z_stp passed to the generator
z_stp = Delta;            % Sampling step size in z -> yields exactly M planes
sigt = 1;             % Total scattering coefficient (unchanged)
lambda = 0.532;       % Wavelength [um] (unchanged)
is_cyclic = 1;        % Cyclic boundary -- see comment above re: GPU memory

spr_params.rad_rng = 0.5;   % Radius range (unchanged)
spr_params.rad_min = 0.2;   % Minimum radius (unchanged)
spr_params.ref_rng = 0.05;  % Refractive index range (unchanged)
spr_params.ref_min = 0.0;   % Minimum refractive index (unchanged)
spr_params.od = 1;          % Optical density (unchanged)
spr_params.scl = 1;         % User scaling factor (unchanged)

%% === Run generator (fixed averaging + extended fine-resolution outputs) ===
[mask_f, mask_f0, z_grid1, mask_f_hr, mask_f0_hr, z_grid1_sps] = tissue_3d_generator( ...
    x_max, x_stp, z_max, z_stp, sigt, spr_params, lambda, is_cyclic);

assert(size(mask_f, 3) == M, 'Expected %d aberration planes, got %d', M, size(mask_f, 3));

% Forward-convention plane positions (z=0 at tissue entrance): plane k is
% placed at k*Delta, so plane M sits exactly at the exit face z=Z_tissue.
plane_z = (1:M)*Delta;   % [1 x M] = [Z_tissue/2, Z_tissue] for M=2

% Full simulated tissue dimensions [width_x, width_y, depth_z], physical units (um).
tissue_dims_um = [2*x_max, 2*x_max, Z_tissue];

%% === Gather from GPU and save for Python ===
mask_f      = gather(mask_f);
mask_f0     = gather(mask_f0);
mask_f_hr   = gather(mask_f_hr);
mask_f0_hr  = gather(mask_f0_hr);
z_grid1     = gather(z_grid1);
z_grid1_sps = gather(z_grid1_sps);

out_file = sprintf('tissue_background_%gx%gx%g.mat', tissue_dims_um(1), tissue_dims_um(2), tissue_dims_um(3));
save(out_file, 'mask_f', 'mask_f0', 'mask_f_hr', 'mask_f0_hr', 'z_grid1', 'z_grid1_sps', ...
    'x_max', 'x_stp', 'z_max', 'M', 'Z_tissue', 'Delta', 'plane_z', 'tissue_dims_um', 'sigt', '-v7.3');

fprintf('Saved outputs to %s (Nplanes = %d, Delta = %.3f, plane_z = %s, tissue_dims_um = %s)\n', ...
    out_file, size(mask_f, 3), Delta, mat2str(plane_z), mat2str(tissue_dims_um));

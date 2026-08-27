% run_tissue_3d_generator_capillary_35x35x15_two_rbcs: Fresh 35x35x15um
% background tissue for the two-RBC scenario (step_two_two_rbcs.py) --
% same generation parameters as run_tissue_3d_generator_capillary_35x35x15.m
% (same tissue size, is_cyclic=1, same spr_params/sigt/lambda), but a NEW
% random sphere realization, saved to a differently-named .mat so it
% doesn't overwrite the original single-RBC scenario's background.

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

%% === Set parameters (identical to the single-RBC 35x35x15 tissue) ===
x_max = 17.5;
x_stp = 0.1;
z_max = 7.5;
M = 2;
Z_tissue = 2*z_max;
Delta = Z_tissue/M;
z_stp = Delta;
sigt = 1;
lambda = 0.532;
is_cyclic = 1;

spr_params.rad_rng = 0.5;
spr_params.rad_min = 0.2;
spr_params.ref_rng = 0.05;
spr_params.ref_min = 0.0;
spr_params.od = 1;
spr_params.scl = 1;

%% === Run generator (fixed averaging + extended fine-resolution outputs) ===
[mask_f, mask_f0, z_grid1, mask_f_hr, mask_f0_hr, z_grid1_sps] = tissue_3d_generator( ...
    x_max, x_stp, z_max, z_stp, sigt, spr_params, lambda, is_cyclic);

assert(size(mask_f, 3) == M, 'Expected %d aberration planes, got %d', M, size(mask_f, 3));

plane_z = (1:M)*Delta;
tissue_dims_um = [2*x_max, 2*x_max, Z_tissue];

%% === Gather from GPU and save for Python ===
mask_f      = gather(mask_f);
mask_f0     = gather(mask_f0);
mask_f_hr   = gather(mask_f_hr);
mask_f0_hr  = gather(mask_f0_hr);
z_grid1     = gather(z_grid1);
z_grid1_sps = gather(z_grid1_sps);

out_file = 'tissue_background_35x35x15_two_rbcs.mat';
save(out_file, 'mask_f', 'mask_f0', 'mask_f_hr', 'mask_f0_hr', 'z_grid1', 'z_grid1_sps', ...
    'x_max', 'x_stp', 'z_max', 'M', 'Z_tissue', 'Delta', 'plane_z', 'tissue_dims_um', 'sigt', '-v7.3');

fprintf('Saved outputs to %s (Nplanes = %d, Delta = %.3f, plane_z = %s, tissue_dims_um = %s)\n', ...
    out_file, size(mask_f, 3), Delta, mat2str(plane_z), mat2str(tissue_dims_um));

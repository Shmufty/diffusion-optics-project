function [mask_f,mask_f0,z_grid1]=tissue_3d_generator(x_max,x_stp,z_max,z_stp,sigt,spr_params,lambda,is_cyclic)
% tissue_3d_generator: Generates 3D phase masks with spherical scatterers
%
% Inputs:
%   x_max: Maximum extent in x/y dimensions [physical units]
%   x_stp: Sampling step size in x/y dimensions [physical units]
%   z_max: Maximum extent in z dimension [physical units]
%   z_stp: Sampling step size in z dimension [physical units]
%   sigt: Total scattering coefficient
%   spr_params: Structure containing sphere parameters (rad_rng, rad_min, ref_rng, ref_min, od, scl)
%   lambda: Wavelength [physical units]
%   is_cyclic: Flag for cyclic boundary conditions (default: 0)
% Outputs:
%   mask_f: Final phase mask [Nx × Nx × Nzs]
%   mask_f0: Phase mask without mean field [Nx × Nx × Nzs]
%   z_grid1: Fine z-grid coordinates [1 × Nz]

% === Initialize parameters and boundary conditions ===
if~exist('is_cyclic','var')
    is_cyclic=0;
end
dim=3;
if ~is_cyclic
    bdr_fact=2;  % Double the working domain for non-cyclic boundaries
else
    bdr_fact=1;  % Use specified domain for cyclic boundaries
end

% === Define working domain extents (larger than output domain) ===
wz_max=z_max*2;%bdr_fact;   % Working z extent: doubled in z
wx_max=x_max*bdr_fact;       % Working x/y extent: doubled for non-cyclic
h_z_stp=z_stp/2;             % Half step in z for grid centering
h_x_stp=x_stp/2;             % Half step in x for grid centering

% === Create 1D coordinate grids ===
% Working grids (larger domain):
wx_grid1=[-wx_max:x_stp:wx_max];                      % [1 × Nxw] Working x/y grid
wz_grid1=[-wz_max+h_x_stp:x_stp:wz_max-h_x_stp];     % [1 × Nzw] Working z grid (fine sampling with x_stp)
% Output grids (smaller domain):
x_grid1=[-x_max:x_stp:x_max];                         % [1 × Nx] Output x/y grid
z_grid1=[-z_max+h_x_stp:x_stp:z_max-h_x_stp];        % [1 × Nz] Output z grid (fine sampling with x_stp)

% === Find indices to extract output region from working domain ===
iix=find((abs(wx_grid1)<=(x_max+x_stp/10)));  % [1 × Nx] Indices for x/y cropping
iiz=find((abs(wz_grid1)<=(z_max+x_stp/10)));  % [1 × Nz] Indices for z cropping

% === Store grid dimensions ===
Nx=length(x_grid1);    % Output grid size in x/y (e.g., 201 for x_max=10, x_stp=0.1)
Nz=length(z_grid1);    % Output grid size in z (fine sampling)
hNx=ceil(Nx/2);        % Half of Nx
Nxw=length(wx_grid1);  % Working grid size in x/y (e.g., 401 for non-cyclic)
Nzw=length(wz_grid1);  % Working grid size in z (e.g., 801 for z_max*4/x_stp)

% === Create 3D meshgrids ===
% Output domain grids:
[x_grid,y_grid,z_grid]=ndgrid(x_grid1,x_grid1,z_grid1);  % Each: [Nx × Nx × Nz]
% Working domain grids:
[wx_grid,wy_grid,wz_grid]=ndgrid(wx_grid1,wx_grid1,wz_grid1);  % Each: [Nxw × Nxw × Nzw]

% === Transfer working grids to GPU for acceleration ===
wx_grid=gpuArray(wx_grid);   % [Nxw × Nxw × Nzw] on GPU
wy_grid=gpuArray(wy_grid);   % [Nxw × Nxw × Nzw] on GPU
wz_grid=gpuArray(wz_grid);   % [Nxw × Nxw × Nzw] on GPU

% === Define frequency domain grids ===
v_max=lambda/x_stp/2;              % Maximum frequency (Nyquist) in x/y
v_stp=lambda/wx_max/2;             % Frequency step in x/y
lpz=lambda/z_stp/2;                % Nyquist frequency in z
v_max=ceil(v_max/v_stp)*v_stp;    % Round v_max to grid
vz_stp=lambda/(wz_max)/2;          % Frequency step in z

% Create 1D frequency grids:
v_grid1=[-v_max:v_stp:v_max];              % [1 × Nvx] Frequency grid in x/y
vz_grid1=[-v_max:vz_stp:v_max-vz_stp/2];  % [1 × Nvz] Frequency grid in z

% Create 3D frequency meshgrids:
[vx_grid,vy_grid,vz_grid]=ndgrid(v_grid1,v_grid1,vz_grid1);  % Each: [Nvx × Nvx × Nvz]

% === Initialize masks ===
mask=ones(Nxw,Nxw,Nzw,'gpuArray');  % [Nxw × Nxw × Nzw] Spatial support mask

% === Create optical transfer function (OTF) mask ===
% Circular aperture in frequency space (NA constraint)
mask_otf=(sqrt(vx_grid.^2+vy_grid.^2)<0.5*v_max);  % [Nvx × Nvx × Nvz] Binary OTF mask

% === Generate random sphere distribution ===
mask_f0=zeros(Nxw,Nxw,Nzw,'gpuArray');  % [Nxw × Nxw × Nzw] Phase accumulation

% Extract sphere parameters:
rad_rng=spr_params.rad_rng;   % Radius range
rad_min=spr_params.rad_min;   % Minimum radius
ref_rng=spr_params.ref_rng;   % Refractive index range
ref_min=spr_params.ref_min;   % Minimum refractive index

% Calculate number of spheres based on optical density:
mean_area=((rad_rng+rad_min)^3-rad_min^3)*pi/3/rad_rng;  % Mean sphere volume
sprN=ceil((spr_params.od)*(2*wx_max)^2/mean_area);       % Number of spheres to generate

% === Loop over each sphere and add its phase contribution ===
for k=1:sprN
    % Generate random sphere properties:
    c=2*(rand(3,1)-0.5).*[wx_max;wx_max;z_max];  % [3 × 1] Center position [x,y,z]
    rad=rand*(rad_rng)+rad_min;                   % Scalar: Sphere radius
    refind=rand*ref_rng+ref_min;                  % Scalar: Refractive index difference
    scmask=zeros(Nxw,Nxw,Nzw,'gpuArray');        % [Nxw × Nxw × Nzw] Single sphere mask

    % === Handle periodic boundary conditions by replicating spheres ===
    for s1=-1:1
        for s2=-1:1
          tc=c; tc(1)=tc(1)+s1*wx_max*2; tc(2)=tc(2)+s2*wx_max*2;  % [3 × 1] Shifted center
          % Compute signed distance from sphere surface:
          cmask=sqrt((wx_grid-tc(1)).^2+(wy_grid-tc(2)).^2+(wz_grid-tc(3)).^2)-rad;  % [Nxw × Nxw × Nzw]
          % Smooth transition at sphere boundary:
          scmask=scmask+((cmask<0)+(cmask>=0).*exp(-cmask/x_stp*1.0));  % [Nxw × Nxw × Nzw]
        end
    end
    % Accumulate phase from this sphere:
    mask_f0=mask_f0+ refind*scmask;  % [Nxw × Nxw × Nzw]
end

% === Convert refractive index to phase ===
mask_f0=exp(2*pi*i*mask_f0);  % [Nxw × Nxw × Nzw] Complex phase mask

% === Apply spatial and frequency domain filters ===
mask_f0_f=fftshift(ifftn(ifftshift(mask_f0)));  % [Nxw × Nxw × Nzw] Transform to real space

% Filter and transform back to frequency domain:
mask_f0=fftshift(fftn(ifftshift(mask_f0_f.*mask)));  % [Nxw × Nxw × Nzw]
mask_f0=mask_f0(iix,iix,iiz);  % [Nx × Nx × Nz] Crop to output domain

% === Normalize mask ===
mask_f0=mask_f0-mean(mask_f0(:));                    % [Nx × Nx × Nz] Remove mean
scl=1/sqrt(mean(abs(mask_f0(:).^2)));               % Scalar: Normalization factor
mask_f0=mask_f0*scl;                                 % [Nx × Nx × Nz] Normalize variance
mask_f0=mask_f0*(1-exp(-sigt*x_stp));               % [Nx × Nx × Nz] Apply scattering coefficient
mask_f0_org=mask_f0;                                 % [Nx × Nx × Nz] Store original

% === Apply OTF filtering ===
tmask_f0_f=fftshift(ifftn(ifftshift(mask_f0)));  % [Nx × Nx × Nz] Transform to real space

mask_f0=fftshift(fftn(ifftshift(mask_f0_f.*mask.*mask_otf)));  % Filter with OTF (size may change based on padding)
mask_f0=mask_f0(iix,iix,iiz);  % [Nx × Nx × Nz] Crop to output domain

% === Final normalization ===
mask_f0=mask_f0-mean(mask_f0(:));         % [Nx × Nx × Nz] Remove mean
mask_f0=mask_f0*scl*spr_params.scl;      % [Nx × Nx × Nz] Apply user scaling
mask_f0=mask_f0*(1-exp(-sigt*x_stp));    % [Nx × Nx × Nz] Apply scattering coefficient

% === Calculate mean field transmission ===
mu=exp(-sigt/2*x_stp);  % Scalar: Mean field attenuation per x_stp

% === Add mean field to get total mask ===
mask_f=mask_f0+mu;  % [Nx × Nx × Nz] Total phase mask (fluctuation + mean)

% === Downsample in z-direction from fine to coarse grid ===
smp_z=z_stp/x_stp;  % Scalar: Z-downsampling ratio (e.g., 5 if z_stp=0.5, x_stp=0.1)

% Create coarse z-grid (using z_stp instead of x_stp):
z_grid1_sps=[-z_max+h_z_stp:z_stp:z_max-h_z_stp];  % [1 × Nzs] Coarse z-grid

Nzs=length(z_grid1_sps);  % Number of coarse z-layers (e.g., 41 vs 201 fine layers)

% === DEBUG: Print grid information ===
fprintf('\n=== Grid Configuration ===\n');
fprintf('x_stp = %.6f\n', x_stp);
fprintf('z_stp = %.6f\n', z_stp);
fprintf('z_max = %.6f\n', z_max);
fprintf('h_z_stp = %.6f\n', h_z_stp);
fprintf('h_x_stp = %.6f\n', h_x_stp);
fprintf('smp_z (z_stp/x_stp) = %.6f\n', smp_z);
fprintf('\nFine z-grid (using x_stp):');
fprintf('\n  z_grid1 range: [%.3f, %.3f]\n', min(z_grid1), max(z_grid1));
fprintf('  z_grid1 step: %.6f\n', z_grid1(2)-z_grid1(1));
fprintf('  Nz (fine layers) = %d\n', Nz);
fprintf('\nCoarse z-grid (using z_stp):');
fprintf('\n  z_grid1_sps range: [%.3f, %.3f]\n', min(z_grid1_sps), max(z_grid1_sps));
fprintf('  z_grid1_sps step: %.6f\n', z_grid1_sps(2)-z_grid1_sps(1));
fprintf('  Nzs (coarse layers) = %d\n', Nzs);
fprintf('==========================\n\n');

% === Initialize downsampled masks ===
mask_f_hr=mask_f;    % [Nx × Nx × Nz] High-resolution mask (backup)
mask_f0_hr=mask_f0;  % [Nx × Nx × Nz] High-resolution fluctuation mask (backup)
mask_f=zeros(Nx,Nx,Nzs,'gpuArray');       % [Nx × Nx × Nzs] Downsampled total mask
mask_f0=zeros(Nx,Nx,Nzs,'gpuArray');      % [Nx × Nx × Nzs] Downsampled fluctuation mask
mask_f0_hr_t=zeros(Nx,Nx,Nz,'gpuArray');  % [Nx × Nx × Nz] Temporary scaled mask

fprintf('=== Downsampling Loop ===\n');

% === Loop over coarse z-layers and average fine layers ===
for j=1:Nzs
    % Find fine z-layers that correspond to this coarse layer:
    ii0=find(abs(z_grid1-z_grid1_sps(j))<x_stp*0.75);  % [1 × Nfine] Indices (typically smp_z indices)
    
    % Debug for first few iterations
    if j <= 3 || isempty(ii0)
        fprintf('\nIteration j=%d/%d:\n', j, Nzs);
        fprintf('  Target z-position: %.6f\n', z_grid1_sps(j));
        fprintf('  Search tolerance: %.6f (x_stp*0.75)\n', x_stp*0.75);
        fprintf('  Found indices ii0: %s\n', mat2str(ii0));
        fprintf('  Number of matches: %d\n', length(ii0));
        
        if ~isempty(ii0)
            fprintf('  Matched z-values: [');
            for idx = ii0
                fprintf('%.3f ', z_grid1(idx));
            end
            fprintf(']\n');
            fprintf('  mask_f0_hr(:,:,ii0) size: [%d, %d, %d]\n', ...
                Nx, Nx, length(ii0));
            fprintf('  mask_f0(:,:,j) target size: [%d, %d]\n', Nx, Nx);
        else
            warning('  ⚠ No matching z-slices found!');
        end
    end
    
    if isempty(ii0)
        warning('No matching fine z-slices for coarse layer j=%d (z=%.3f)', j, z_grid1_sps(j));
        continue;
    end
    
    % === Perform downsampling by averaging ===
    try
        % Average fine layers and scale by sampling ratio:
        mask_f0(:,:,j) = mean(mask_f0_hr(:,:,ii0), 3) * smp_z;  % [Nx × Nx × 1] Single coarse layer
        % Add mean field (scaled by smp_z):
        mask_f(:,:,j) = mask_f0(:,:,j) + mu^smp_z;              % [Nx × Nx × 1] Single coarse layer
        % Store scaled fine layers:
        mask_f0_hr_t(:,:,ii0) = mask_f0_hr(:,:,ii0) * smp_z;   % [Nx × Nx × Nfine] Scaled fine layers
    catch ME
        fprintf('\n✗ ERROR at j=%d:\n', j);
        fprintf('  mask_f0(:,:,j) target: [%d, %d]\n', size(mask_f0, 1), size(mask_f0, 2));
        fprintf('  mask_f0_hr(:,:,ii0) source: [%d, %d, %d]\n', ...
            size(mask_f0_hr(:,:,ii0), 1), size(mask_f0_hr(:,:,ii0), 2), size(mask_f0_hr(:,:,ii0), 3));
        fprintf('  mean(..., 3) result: [%d, %d]\n', ...
            size(mean(mask_f0_hr(:,:,ii0), 3), 1), size(mean(mask_f0_hr(:,:,ii0), 3), 2));
        rethrow(ME);
    end
end
fprintf('=========================\n\n');
% Output dimensions:
% mask_f: [Nx × Nx × Nzs] - Downsampled total phase mask
% mask_f0: [Nx × Nx × Nzs] - Downsampled fluctuation phase mask
% z_grid1: [1 × Nz] - Fine z-grid coordinates


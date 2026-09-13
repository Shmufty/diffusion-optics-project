% This MATLAB code simulates light propagation in a scattering and absorbing 
% medium and reconstructs the positions of multiple light sources. It generates
% random source positions, calculates the resulting radiance, and optimizes the 
% reconstruction of source coordinates using polynomial fitting and the 
% Lambert W function. The results are visualized through radiance and signal plots.

% toolboxes needed:
% Optimization Toolbox: For polynomial fitting.
% Symbolic Math Toolbox: For the Lambert W function.
%% 
clc;
clear;
close all;
%% Parameters
N = 101; % Number of points in each dimension of the grid
mua = 0.1; % Absorption coefficient
mus = 1000; % Scattering coefficient
g = 0.9; % Anisotropy factor
dx = 0.01; % Grid spacing
num_sources = 3; % Number of light sources
M_threshold = 5; % Threshold

%% Derived parameters
musp = mus * (1 - g); % Reduced scattering coefficient
D = 1 / (3 * (mua + musp)); % Diffusion coefficient
mu_eff = sqrt(mua / D); % Effective attenuation coefficient
n = N / 2 - 0.5; % Half the grid size 
x = (-n:n) * dx; % X coordinates
z = (0:N-1) * dx; % Z coordinates

% Create a meshgrid for the x and z coordinates
[X, Z] = meshgrid(x, z);

% Constants
const = 1 / (4 * pi * D); % Constant used in calculations

%% Source positions
% Randomly generate source positions within the grid
x0 = rand(num_sources, 1) * 2 * x(end - 5) - x(end - 5);
z0 = rand(num_sources, 1) * z(40) + dx * 5;

% Initialize radiance
rad = 0;

% Calculate radiance for each source
for i_s = 1:num_sources
    [~, this_r] = cart2pol(X - x0(i_s), Z - z0(i_s));
    rad = rad + const * (exp(-mu_eff * this_r)) ./ (this_r);
    rad(isinf(rad)) = max(rad(~isinf(rad)), [], 'all');
end
I = rad(1, :);

%% Optimization to reconstruct source positions
I_hat = I;
Ri = 0;
n_vic = 3; % Vicinity range for polynomial fitting
pol_x = (-n_vic:n_vic) * dx;
H = [pol_x'.^2 pol_x' ones(size(pol_x'))]; % Polynomial fitting matrix

% Initialize variables
flag = 1;
i_s = 1;

while (flag == 1)
    [M, maxind] = max(I_hat);
    if (M < M_threshold || maxind < 5 || maxind > N - 5)
        break;
    end
    
    try
        % Polynomial fitting to find maximum
        y = I_hat(maxind - n_vic : maxind + n_vic)';
        pol = H \ y;
        a = pol(1);
        b = pol(2);
        c = pol(3);
        M = c - b^2 / (4 * a);
        x0_hat(i_s) = -b / (2 * a) + x(maxind);
    catch
        x0_hat(i_s) = x(maxind);
    end
    % Estimate z0 using Lambert W function
    W_arg = mu_eff * const / M;
    z0_hat(i_s) = lambertw(W_arg) / mu_eff;

    % Update the estimated radiance
    [~, r_hat] = cart2pol(X - x0_hat(i_s), Z - z0_hat(i_s));
    Ri = const * (exp(-mu_eff * r_hat)) ./ (r_hat);
    I_hat = I_hat - Ri(1, :);
    i_s = i_s + 1;
end

% Remove bad estimations
bads = find(z0_hat>(N-1)*dx);
z0_hat(bads) = [];
x0_hat(bads) = [];

%% Plotting results
figure;
set(gcf, 'Position', [488, 182.2, 753.8, 579.8]);

subplot(2, 2, 2);
imagesc(x, z, rad);
hold on;
plot(x0_hat, z0_hat, '*r');
daspect([1 1 1]);
title('Restoration');
xlabel('x');
ylabel('z');

subplot(2, 2, 1);
imagesc(x, z, log10(rad));
xlabel('x');
ylabel('z');
daspect([1 1 1]);
title('Tissue radiance (log)');

subplot(2, 2, [3 4]);
plot(x, I, 'LineWidth', 3);
xlabel('x');
title('Signal');


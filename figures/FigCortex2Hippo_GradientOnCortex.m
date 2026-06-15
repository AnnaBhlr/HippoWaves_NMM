%% Plot principal gradient on fsaverage5 left hemisphere mesh

% --- paths (adjust if needed) ---
repoRoot = fileparts(fileparts(mfilename('fullpath')));  % repo root (data/ paths)
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/slanCM'));
addpath(genpath('/Users/ab799/Documents/Hippocampus/meshes/'));  % where fsaverage5_hemi_L_pial_noMedialWall lives

% --- 1) load gradient values ---
gradFile = fullfile(repoRoot, 'resampling_principal_gradient', 'resampled_principal_gradient.csv');
g = readmatrix(gradFile);

% Ensure it's a column vector
g = g(:);

% --- 2) load mesh ---
fsaverage5_hemi_L_pial_noMedialWall;   % this should create msh
hemiVertices = msh.POS;                 % mm
hemiFaces    = msh.TRIANGLES(:, 1:3);

% --- sanity checks ---
nV = size(hemiVertices, 1);
if numel(g) ~= nV
    error('Gradient length (%d) does not match #vertices in mesh (%d).', numel(g), nV);
end

% Optional: handle NaNs (e.g., medial wall already removed, but just in case)
% g(isnan(g)) = 0;
%%
% --- 3) plot gradient onto mesh ---
colorscheme  = 'YlOrBr';                % pick any ScientificColourMap name you like
my_colormap  = flipud(slanCM(colorscheme));

figure;
figure_surf2(hemiFaces, hemiVertices, g, my_colormap, -130, 15);
hold on;

% overlay edges
patch('faces', hemiFaces, 'vertices', hemiVertices, ...
      'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);

% color limits: use data-driven range (recommended)
clim(prctile(g, [1 99]));   % or clim([min(g) max(g)]) if you prefer

% save
outFile = '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/principal_gradient_cortex_flipped.png';
print('-dpng', '-r300', outFile);

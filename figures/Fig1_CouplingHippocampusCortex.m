repoRoot = fileparts(fileparts(mfilename('fullpath')));  % repo root (data/ paths)
addpath(genpath('/Volumes/entities/research/NEWYSNG/Anna/MATLAB/neural-flows-master'))
addpath(genpath('/Volumes/entities/research/NEWYSNG/Anna/MATLAB/ScientificColormaps'))
addpath(genpath('/Volumes/entities/research/NEWYSNG/Anna/MATLAB/slanCM'))
addpath(genpath('/Volumes/entities/research/NEWYSNG/Anna/MATLAB/Hippocampus'))

az = -130;
el = 15;
% Load meshes
hippocampus;
hippoVertices = msh.POS; % in mm
hippoFaces = msh.TRIANGLES(:,1:3);
hippoNnodes = size(hippoVertices,1);

fsaverage5_hemi_L_pial_noMedialWall;
hemiVertices = msh.POS; % in mm
hemiFaces = msh.TRIANGLES(:,1:3);
clear msh;

% Load CSV file with indices
%connection_indices = csvread(fullfile(repoRoot, 'data', 'coupling', 'hemi2hippo.csv')); % Assuming 1-based indices
connection_indices = csvread(fullfile(repoRoot, 'data', 'coupling', 'hemi2hippo_random.csv'));
%%
% Ensure indices are valid (integers, positive, within range)
connection_indices = round(connection_indices); % Round to nearest integer
connection_indices = connection_indices(~isnan(connection_indices)); % Remove NaNs
connection_indices = connection_indices(connection_indices > 0 & connection_indices <= hippoNnodes); % Valid range

% Assign colors to hippocampus vertices based on their index
hippoColors = parula(hippoNnodes); % Generate colormap
hippoVertexColors = hippoColors;   % Assign each vertex a color

% Initialize hemisphere colors as gray
hemiVertexColors = ones(size(hemiVertices,1), 3) * 1;

% Assign the same color to connected hemisphere vertices
for i = 1:length(connection_indices)
    hemiVertexColors(i, :) = hippoColors(connection_indices(i), :);
end

%% Plot Hippocampus
figure;
scatter3(hippoVertices(:,1), hippoVertices(:,2), hippoVertices(:,3), 20, hippoVertexColors, 'filled');
hold on;
patch('faces', hippoFaces, 'vertices', hippoVertices, 'facecolor', [1 1 1], 'edgecolor', [0.5 0.5 0.5]);
view(az, el);
axis equal; axis off; grid off;
title('Hippocampus');

% Save figure as PNG at 300 DPI
%print('-dpng', '-r300', '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/hippo_coupling.png');
%%
%Plot Hemisphere
figure;
scatter3(hemiVertices(:,1), hemiVertices(:,2), hemiVertices(:,3), 40, hemiVertexColors, 'filled');
hold on;
patch('faces', hemiFaces, 'vertices', hemiVertices, 'facecolor', [1 1 1], 'edgecolor', [0.5 0.5 0.5]);
view(az, el);
axis equal; axis off; grid off;
title('Hemisphere');

% Save figure as PNG at 300 DPI
%print('-dpng', '-r300', '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/cortex_coupling_random.png');
%% Paths (macOS)
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/neural-flows-master'))
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/ScientificColormaps'))
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/slanCM'))
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/Hippocampus'))

%az = -130;  el = 15;
az = 100; el = 0; %lateral view

% --- Load meshes ---
% Hippocampus
hippocampus;
hippoVertices = msh.POS;                    % (Nhip x 3), mm
hippoFaces    = msh.TRIANGLES(:,1:3);       % (Fhip x 3), 1-based indices
hippoNnodes   = size(hippoVertices,1);

% Hemisphere (fsaverage5 L pial, no medial wall)
fsaverage5_hemi_L_pial_noMedialWall;
hemiVertices = msh.POS;                     % (Nhemi x 3), mm
hemiFaces    = msh.TRIANGLES(:,1:3);        % (Fhemi x 3), 1-based
hemiNnodes   = size(hemiVertices,1);
clear msh;

% --- Read mapping: hemi-vertex -> hippo-vertex index ---
% CSV contains one column, length = 9204, with integers or the string "nan".
mapPath = fullfile(repoRoot, 'data', 'coupling', 'hemi2hippo.csv');
%mapPath = fullfile(repoRoot, 'data', 'coupling', 'hemi2hippo_random.csv');
connection_indices = readmatrix(mapPath, 'OutputType','double');   % "nan" -> NaN
connection_indices = connection_indices(:);

% Basic sanity checks
if numel(connection_indices) ~= hemiNnodes
    warning('Mapping length (%d) != #hemi vertices (%d). Will use min length.', ...
        numel(connection_indices), hemiNnodes);
    L = min(numel(connection_indices), hemiNnodes);
    connection_indices = connection_indices(1:L);
end

% Handle 0-based vs 1-based: if max non-NaN equals hippoNnodes-1, treat as 0-based
nonan = connection_indices(~isnan(connection_indices));
if ~isempty(nonan) && max(nonan) <= hippoNnodes-1 && any(nonan==0)
    connection_indices = connection_indices + 1;   % shift to 1-based
end

% Clamp to valid range and keep NaNs
connection_indices(~isnan(connection_indices) & (connection_indices < 1 | connection_indices > hippoNnodes)) = NaN;

% --- Define scalar values on hippocampus vertices ---
% Here: simple example = vertex index (you can replace with your real data)
hippoScalar = (1:hippoNnodes)';             % (Nhip x 1)
% Normalise to [0,1] for prettier colour scaling (optional)
hippoScalarNorm = (hippoScalar - min(hippoScalar)) / (max(hippoScalar) - min(hippoScalar) + eps);

% Hemisphere scalar = mapped from hippocampus; NaN where no mapping
hemiScalar = nan(hemiNnodes,1);
valid = ~isnan(connection_indices);
hemiScalar(valid) = hippoScalarNorm(connection_indices(valid));

% --- Choose a colormap ---

cmap = parula(256);

% --- Plot Hippocampus with trisurf ---

% hemiScalar: [Nhemi x 1], NaN where unmapped
hemiScalar_filled = mesh_harmonic_inpaint(hemiVertices, hemiFaces, hemiScalar);

% Optional mild smoothing (one or two iterations) to remove residual seams:
TR = triangulation(hemiFaces, hemiVertices);
E  = TR.edges; N = size(hemiVertices,1);
A  = sparse([E(:,1);E(:,2)], [E(:,2);E(:,1)], 1, N, N);
D  = spdiags(sum(A,2), 0, N, N);
L  = D - A;

alpha = 0.15; nIter = 3; vals = hemiScalar_filled;
for k = 1:nIter, vals = vals - alpha * L * vals; end
hemiScalar_smooth = vals;

% Plot with trisurf (smooth, continuous colouring)
figure('Color','w');


f1 = figure('Color','w'); 
trisurf(hippoFaces, hippoVertices(:,1), hippoVertices(:,2), hippoVertices(:,3), hippoScalarNorm, ...
        'EdgeColor','none', 'FaceColor','interp');
axis equal off; view(az, el);
colormap(f1, cmap);
caxis([0 1]); colorbar; title('Hippocampus');
%camlight headlight; lighting gouraud; material dull;

% Save 300-DPI PNG (macOS-friendly)
exportgraphics(f1, '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/hippo_coupling_trisurf_lateral.png', 'Resolution', 300);

% --- Plot Hemisphere with trisurf (mapped colours) ---
f2 = figure('Color','w');
trisurf(hemiFaces, hemiVertices(:,1), hemiVertices(:,2), hemiVertices(:,3), ...
        hemiScalar_smooth, 'EdgeColor','none','FaceColor','interp');
axis equal off; view(az,el); lighting gouraud; camlight headlight; material dull;
colormap(parula(256)); colorbar; % swap to your fav ScientificColormap if you like

exportgraphics(f2, '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/hemi_coupling_trisurf_lateral.png', 'Resolution', 300);



function vals_full = mesh_harmonic_inpaint(vertices, faces, vals_known)
% Fill NaNs on a triangular mesh by harmonic extension (Laplace equation).
% Keeps known values fixed; fills unknowns smoothly.
% vertices: [N x 3], faces: [F x 3], vals_known: [N x 1] with NaNs for unknowns

N  = size(vertices,1);
TR = triangulation(faces, vertices);

% Build symmetric vertex adjacency
E = TR.edges;                                          % [M x 2]
A = sparse([E(:,1); E(:,2)], [E(:,2); E(:,1)], 1, N, N);
D = spdiags(sum(A,2), 0, N, N);
L = D - A;                                             % unnormalised graph Laplacian

vals_full = vals_known(:);
K = ~isnan(vals_full);                                 % known mask
U = ~K;                                                % unknown mask
if ~any(U); return; end

% Partition L into known/unknown blocks and solve L_uu x_u = -L_uk x_k
Luu = L(U,U);
Luk = L(U,K);
xk  = vals_full(K);

xu  = - (Luu \ (Luk * xk));                            % solve
vals_full(U) = xu;
end

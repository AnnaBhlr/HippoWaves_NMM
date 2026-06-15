% Add necessary paths
repoRoot = fileparts(fileparts(mfilename('fullpath')));  % repo root (data/ paths)
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/brainwaves-master'));
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/neural-flows-master'));
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/ScientificColormaps'));
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/Hippocampus'));
addpath(genpath('/Users/ab799/Documents/MATLAB/cnem_25-05-23' ))
addpath(genpath('/Users/ab799/Documents/Hippocampus/meshes/'));
%%
outputFolder = '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/Cortex2Hippo';

simulations = {...
    '/Volumes/NEWYSNG/Anna/HippocampalCorticalModeling/cortex2hippo/Output_Hippowaves_slow_neg10_010226/20260204_215244', ... 
    '/Volumes/NEWYSNG/Anna/HippocampalCorticalModeling/cortex2hippo/Output_Hippowaves_slow_pos10_010226/20260204_162512'...
    };


timesToPlot = [2,5,6,6.5, 7,7.5, 8,10];
timesToPlot = [5.25,6.75, 7.3,7.8, 8.5];
timesToPlot = [6.75];

timesToPlot = [1.5, 13];


colormap_main = readmatrix(fullfile(repoRoot, 'data', 'colormaps', 'RdYlBu_colormap.csv'));
colormap_main = flipud(colormap_main); % Flip to match the desired color scheme

% get meshes
hippocampus;
hippoVertices = msh.POS; % in mm
hippoFaces = msh.TRIANGLES(:, 1:3);
hippoNnodes = size(hippoVertices, 1);
%g
% --- spatial smoothing operator ---
L = mesh_laplacian_uniform(hippoFaces, hippoNnodes);

tau = 0.2;          % smoothing strength per step (try 0.05–0.5)
nSteps = 5;         % number of diffusion steps (try 1–10)

S = speye(hippoNnodes) + tau * L;   % implicit diffusion step
% Optional: factor once for speed (works if S is SPD)
[R,p] = chol(S,'lower');
useChol = (p == 0);

%
for k = 1:numel(simulations)

    thisFolder = simulations{k};          % full path to the simulation folder
    [~, runName] = fileparts(thisFolder); % e.g., '20250919_140940'

    % time vector path: .../time_vector_<runName>.csv
    inputDataTime = fullfile(thisFolder, sprintf('time_vector_%s.csv', runName));
    time = readmatrix(inputDataTime);
    inputDataFieldAct = fullfile(thisFolder, sprintf('fieldact_%s.csv', runName));
    fieldAct = readmatrix(inputDataFieldAct);

    relaxTime = 0.05; % seconds

        % discard relaxtion time
    dt = mean(diff(time)); % time step in s 
    discardIdx = find(time >= relaxTime, 1, 'first');
    if ~isempty(discardIdx)
        time = time(discardIdx:end) - time(discardIdx);
        fieldAct =fieldAct(discardIdx:end, 1:hippoNnodes);
    end

    % demean each node's time series (same as your bsxfun step)
    fieldAct0 = bsxfun(@minus, fieldAct, mean(fieldAct, 1));

    % analytic signal per node (Hilbert along time dimension)
    analytic = hilbert(fieldAct0);          % size: T x N, complex

    % spatial smoothing in node dimension (solve N x T system)
    Y = analytic.';                         % N x T
    for s = 1:nSteps
        if useChol
            Y = R'\(R\Y);                   % solves (I+tauL) * Ynew = Y
        else
            Y = S \ Y;
        end
    end

    analytic_smooth = Y.';                  % back to T x N

    % (optional but helpful) normalise to unit phasor before angle
    phasor = analytic_smooth ./ max(abs(analytic_smooth), eps);

    % unwrap in time per node
    yphaseHippo = unwrap(angle(phasor), [], 1);
    
    yphaseHippo_raw = unwrap(angle(hilbert(bsxfun(@minus,fieldAct,mean(fieldAct)))));

    % velocity
    vHippo = phaseflow_cnem(yphaseHippo, hippoVertices, dt*1000, 5); % full vector field
    
    t0 = find(time >= 6, 1, 'first');
    figure;
    subplot(1,2,1); trisurf(hippoFaces, hippoVertices(:,1), hippoVertices(:,2), hippoVertices(:,3), fieldAct0(t0,:)); shading interp; title('raw signal');
    subplot(1,2,2); trisurf(hippoFaces, hippoVertices(:,1), hippoVertices(:,2), hippoVertices(:,3), real(analytic_smooth(t0,:))); shading interp; title('smoothed signal');

    str = simulations{k};
    % Pattern: find 8 digits, an underscore, then 6 digits
    pattern = '\d{8}_\d{6}';
    % Extract the match
    match = regexp(str, pattern, 'match');
    lastBit = match{1};

    %yphaseHippo=unwrap(angle(hilbert(bsxfun(@minus,fieldAct,mean(fieldAct)))));
    %vHippo = phaseflow_cnem(yphaseHippo,hippoVertices,dt*1000, 0); % in mm/ms

    % .mat path: .../<runName>_v.mat
    %thisFilePath = fullfile(thisFolder, sprintf('simulation_%s.mat', runName));
    %load(thisFilePath); % expects hippoFaces, hippoVertices, vHippo, etc.

    for i = 1:numel(timesToPlot)
        idxTime = find(time >= timesToPlot(i), 1, 'first');

        f = figure;
        patch('faces', hippoFaces, 'vertices', hippoVertices, ...
              'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
        hold on;
        plot_vectors_on_mesh(hippoFaces, hippoVertices, vHippo, idxTime, -130, 15, [4 4]);

        figName = sprintf('%s_VectorField_Time_%.2f_s.png', runName, double(timesToPlot(i)));
        savePath = fullfile(outputFolder, figName);

        print(f, savePath, '-dpng', '-r300');
        close(f);


 
        % Plot and save hemisphere mesh
        figure;
        set(gcf, 'Position', [100, 100, 900, 500]);
        figure_surf2(hippoFaces, hippoVertices,fieldAct(idxTime, :), colormap_main, az, el);
        %patch('faces', hemiFaces, 'vertices', hemiVertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
        title(sprintf('Dataset %s at t = %.2f s', simulations{datasetIdx}, time(idxTime)));
        colorbar;

        clim([min(min(fieldAct(:, :))), ...
        max(max(fieldAct(:, :)))]);

        hemiFilename = fullfile(outputFolder, sprintf('hippocampus_%s_%.2f.png', lastBit, time(idxTime)));
        %material dull;
        %camlight;
        print('-dpng', '-r300', hemiFilename);
    end
end

function L = mesh_laplacian_uniform(faces, nV)
% Uniform (graph) Laplacian on a triangular mesh.
% L = D - A where A is vertex adjacency (unweighted).

i = [faces(:,1); faces(:,2); faces(:,3); faces(:,1); faces(:,2); faces(:,3)];
j = [faces(:,2); faces(:,3); faces(:,1); faces(:,3); faces(:,1); faces(:,2)];

A = sparse(i, j, 1, nV, nV);
A = spones(A + A');                 % make symmetric, binary
D = spdiags(sum(A,2), 0, nV, nV);
L = D - A;
end
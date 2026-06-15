% Add necessary paths
repoRoot = fileparts(fileparts(mfilename('fullpath')));  % repo root (data/ paths)
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/brainwaves-master'));
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/neural-flows-master'));
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/ScientificColormaps'));
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/Hippocampus'));
addpath(genpath('/Users/ab799/Documents/Hippocampus/meshes/'));
%
outputFolder = '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/Cortex2Hippo';
  
theta = 'slow';
simulations = {...
    '/Volumes/NEWYSNG/Anna/HippocampalCorticalModeling/cortex2hippo/Output_Hippowaves_slow_neg10_010226/20260204_215244', ... 
    '/Volumes/NEWYSNG/Anna/HippocampalCorticalModeling/cortex2hippo/Output_Hippowaves_slow_pos10_010226/20260204_162512'...
    };

% Get meshes
hippocampus;
hippoVertices = msh.POS; % in mm
hippoFaces = msh.TRIANGLES(:, 1:3);
hippoNnodes = size(hippoVertices, 1);

fsaverage5_hemi_L_pial_noMedialWall;
hemiVertices = msh.POS; % in mm
hemiFaces = msh.TRIANGLES(:, 1:3);
clear msh

relaxTime = 0.05; % Relaxation time in seconds
az = -130;
el = 15;

% Define colormap
colormap_main = readmatrix(fullfile(repoRoot, 'data', 'colormaps', 'RdYlBu_colormap.csv'));
colormap_main = flipud(colormap_main); % Flip to match the desired color scheme

fieldact_hippo_all = cell(1, length(simulations));
fieldact_sphere_all = cell(1, length(simulations));
time_all = cell(1, length(simulations));

for datasetIdx = 1:length(simulations)
    inputFolderPath = simulations{datasetIdx}
    csvFilesFolder = dir(fullfile(inputFolderPath, '*.csv'));

    for i = 1:length(csvFilesFolder)
        fileName = csvFilesFolder(i).name;
        fullFilePath = fullfile(csvFilesFolder(i).folder, csvFilesFolder(i).name);
        if startsWith(fileName, 'time')
            time = readmatrix(fullFilePath, 'Delimiter', ',');
        elseif startsWith(fileName, 'fieldact')
            fieldact = readmatrix(fullFilePath, 'Delimiter', ',');
        end
    end

    if isempty(time) || isempty(fieldact)
        error('Time or field activity data could not be loaded.');
    end

    discardIdx = find(time >= relaxTime, 1, 'first');
    if ~isempty(discardIdx)
        time = time(discardIdx:end) - time(discardIdx);
        fieldact_hippo_all{datasetIdx} = fieldact(discardIdx:end, 1:hippoNnodes);
        fieldact_sphere_all{datasetIdx} = fieldact(discardIdx:end, hippoNnodes+1:end);
        time_all{datasetIdx} = time;
    end
end

%%
% Define specific time points you want to capture (actual time values in seconds)

%timePoints = [3.5, 4.2, 6.5, 6.6, 6.7, 6.8, 6.4, 6.2, 5.7, 5.5, 7.5, 7.25];
%timePoints = [3.4, 6.3, 7.2, 6.1,3.45, 6.15];
timePoints = [10.25,10.5, 10.71, 11, 11.5, 11.75, 12.25];
% Initialize storage for order parameter data
all_RHemi = cell(1, length(simulations));
all_RHippo = cell(1, length(simulations));

for datasetIdx = 1:length(simulations)

    fieldact_mesh = fieldact_hippo_all{datasetIdx};
    fieldact_sphere = fieldact_sphere_all{datasetIdx};
    time = time_all{datasetIdx};

    % Compute phase and order parameter
    yphase = unwrap(angle(hilbert(bsxfun(@minus,fieldact_sphere,mean(fieldact_sphere)))));
    R = abs(mean(exp(1i * yphase), 2)); % Coherence 'order parameter'
    % Store R values
    all_RHemi{datasetIdx} = R;

    yphase = unwrap(angle(hilbert(bsxfun(@minus,fieldact_mesh,mean(fieldact_mesh)))));
    R = abs(mean(exp(1i * yphase), 2)); % Coherence 'order parameter'
    % Store R values
    all_RHippo{datasetIdx} = R;

    % Find the indices corresponding to the specified time values
    indices = arrayfun(@(t) find(abs(time - t) == min(abs(time - t)), 1), timePoints);

    str = simulations{datasetIdx};
    % Pattern: find 8 digits, an underscore, then 6 digits
    pattern = '\d{8}_\d{6}';
    % Extract the match
    match = regexp(str, pattern, 'match');
    lastBit = match{1};


    % Generate and save plots
    for idx = 1:length(indices)
        i = indices(idx); % Get the index corresponding to the specific time value

        % Extract the data for this time point
        dataVidHippo = fieldact_mesh(i, :);
        dataVidHemi = fieldact_sphere(i, :);

        % Plot and save hippocampus mesh
        % figure;
        % set(gcf, 'Position', [100, 100, 900, 500]);
        % figure_surf2(hippoFaces, hippoVertices, dataVidHippo, colormap_main, az, el);
        % %patch('faces', hippoFaces, 'vertices', hippoVertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
        % title(sprintf('Dataset %s at t = %.2f s', simulations{datasetIdx}, time(i)));
        % colorbar;
        % clim([min(min(fieldact_mesh(200:end, :))), max(max(fieldact_mesh(200:end, :)))]);
        % [~, lastBit] = fileparts(simulations{datasetIdx});
        % hippoFilename = fullfile(outputFolder, sprintf('hippocampus_%s_%.2f.png', lastBit, time(i)));
        % print('-dpng', '-r300', hippoFilename);
        % close;
    % 
        % Plot and save hemisphere mesh
        figure;
        set(gcf, 'Position', [100, 100, 900, 500]);
        figure_surf2(hemiFaces, hemiVertices, dataVidHemi, colormap_main, az, el);
        %patch('faces', hemiFaces, 'vertices', hemiVertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
        title(sprintf('Dataset %s at t = %.2f s', simulations{datasetIdx}, time(i)));
        colorbar;

        clim([min(min(fieldact_sphere_all{datasetIdx}(200:end, :))), ...
        max(max(fieldact_sphere_all{datasetIdx}(200:end, :)))]);

        hemiFilename = fullfile(outputFolder, sprintf('hemisphere_%s_%.2f.png', lastBit, time(i)));
        %material dull;
        %camlight;
        print('-dpng', '-r300', hemiFilename);
        close;
    end
end

%%
% Common axis limits across datasets
xMin = min(cellfun(@min, time_all));
xMax = max(cellfun(@max, time_all));

for datasetIdx = 1:numel(simulations)

    f = figure;
    set(f, 'Position', [100, 100, 900, 400]);
    hold on;

    y = smoothdata(all_RHemi{datasetIdx}, "gaussian", 30);

    plot(time_all{datasetIdx}, y, 'LineWidth', 1.5, 'Color', 'k');

    xlabel('Time (s)');
    ylabel('Coherence (R)');
    xlim([0, 15]);
    xticks(0:1:15);
    ylim([0, 1.05]);
    box on;

    % Use just the last folder name for nicer filenames/titles
    [~, dsName] = fileparts(simulations{datasetIdx});
    title(sprintf('%s (%s)', dsName, theta), 'Interpreter', 'none');

    % Save
    pngName = fullfile(outputFolder, sprintf('order_parameter_R_Hemi_%s_%s.png', theta, dsName));
    print(f, '-dpng', '-r300', pngName);

    svgName = fullfile(outputFolder, sprintf('order_parameter_R_Hemi_%s_%s.svg', theta, dsName));
    print(f, svgName, '-dsvg');

    close(f);

    f = figure;
    set(f, 'Position', [100, 100, 900, 400]);
    hold on;

    y = smoothdata(all_RHippo{datasetIdx}, "gaussian", 30);

    plot(time_all{datasetIdx}, y, 'LineWidth', 1.5, 'Color', 'k');

    xlabel('Time (s)');
    ylabel('Coherence (R)');
    xlim([0, 15]);
    xticks(0:1:15);
    ylim([0, 1.05]);
    box on;

    % Use just the last folder name for nicer filenames/titles
    [~, dsName] = fileparts(simulations{datasetIdx});
    title(sprintf('%s (%s)', dsName, theta), 'Interpreter', 'none');

    % Save
    pngName = fullfile(outputFolder, sprintf('order_parameter_R_Hippo_%s_%s.png', theta, dsName));
    print(f, '-dpng', '-r300', pngName);

    svgName = fullfile(outputFolder, sprintf('order_parameter_R_Hippo_%s_%s.svg', theta, dsName));
    print(f, svgName, '-dsvg');

    close(f);

end


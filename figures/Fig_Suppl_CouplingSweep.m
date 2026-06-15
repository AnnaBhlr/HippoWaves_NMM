%% Add required paths
addpath(genpath('/Volumes/entities/research/NEWYSNG/Anna/MATLAB/brainwaves-master'));
addpath(genpath('/Volumes/entities/research/NEWYSNG/Anna/MATLAB/neural-flows-master'));
addpath(genpath('/Volumes/entities/research/NEWYSNG/Anna/MATLAB/Hippocampus'));

%% Settings

parentFolders = { ...
    '/Volumes/DATA/Output_Hippowaves_slow_posP_h2hVar', ...
    '/Volumes/DATA/Output_Hippowaves_slow_negP_h2hVar' ...
};

outputFolder = '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/R_grid';

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

relaxTime = 0.05;      % seconds to discard
smoothWindow = 30;     % Gaussian smoothing window

%% Get number of hippocampal nodes
hippocampus;
hippoNnodes = size(msh.POS, 1);
clear msh

%% Find timestamped subfolders and sort them ascending
allSubfolders = cell(1, numel(parentFolders));

for p = 1:numel(parentFolders)

    d = dir(parentFolders{p});
    d = d([d.isdir]);
    names = {d.name};
    names = names(~ismember(names, {'.', '..'}));

    % Keep folders that look like timestamps: YYYYMMDD_HHMMSS
    isTimestamp = ~cellfun(@isempty, regexp(names, '^\d{8}_\d{6}$', 'once'));
    names = names(isTimestamp);

    % Sort ascending by timestamp
    names = sort(names);

    if numel(names) ~= 5
        warning('Expected 5 timestamped folders in %s, found %d.', parentFolders{p}, numel(names));
    end

    allSubfolders{p} = names;
end

% Load data and compute hippocampal R
R_all = cell(5, 2);
time_all = cell(5, 2);
label_all = cell(5, 2);

for col = 1:2

    parentFolder = parentFolders{col};
    simulations = allSubfolders{col};

    for row = 1:numel(simulations)

        simName = simulations{row};
        inputFolderPath = fullfile(parentFolder, simName);

        csvFiles = dir(fullfile(inputFolderPath, '*.csv'));

        time = [];
        fieldact = [];

        for i = 1:numel(csvFiles)
            fileName = csvFiles(i).name;
            fullFilePath = fullfile(csvFiles(i).folder, fileName);

            if startsWith(fileName, 'time')
                time = readmatrix(fullFilePath, 'Delimiter', ',');
            elseif startsWith(fileName, 'fieldact')
                fieldact = readmatrix(fullFilePath, 'Delimiter', ',');
            end
        end

        if isempty(time) || isempty(fieldact)
            error('Missing time or fieldact file in: %s', inputFolderPath);
        end

        % Discard relaxation period
        discardIdx = find(time >= relaxTime, 1, 'first');
        if isempty(discardIdx)
            error('No time point >= relaxTime found in: %s', inputFolderPath);
        end

        time = time(discardIdx:end) - time(discardIdx);
        fieldact = fieldact(discardIdx:end, :);

        % HIPPOCAMPUS ONLY: first hippoNnodes columns
        fieldact_hippo = fieldact(:, 1:hippoNnodes);

        % Compute phase and order parameter R for hippocampus
        yphase_hippo = unwrap(angle(hilbert(bsxfun(@minus, ...
            fieldact_hippo, mean(fieldact_hippo)))));

        R_hippo = abs(mean(exp(1i * yphase_hippo), 2));

        % Store
        time_all{row, col} = time;
        R_all{row, col} = R_hippo;
        label_all{row, col} = simName;
    end
end

%% Plot 5x2 grid
% Each column = one parent folder
% Within each column: top to bottom sorted by timestamp

f = figure;
set(f, 'Position', [100, 100, 1100, 1400]);

tiledlayout(5, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for row = 1:5
    for col = 1:2

        nexttile;

        time = time_all{row, col};
        R = R_all{row, col};

        if isempty(time) || isempty(R)
            title(sprintf('Missing dataset %d,%d', row, col));
            axis off;
            continue;
        end

        plot(time, smoothdata(R, "gaussian", smoothWindow), 'LineWidth', 1.5,'Color', 'k');

        ylim([0, 1.05]);
        xlim([min(time), max(time)]);
        box on;

        title(strrep(label_all{row, col}, '_', '\_'), 'Interpreter', 'tex');

        if row == 5
            xlabel('Time (s)');
        end

        if col == 1
            ylabel('Hippocampal coherence R');
        end
    end
end

sgtitle('Hippocampal order parameter R');

% Save
pngName = fullfile(outputFolder, 'hippocampal_order_parameter_R_5x2.png');
svgName = fullfile(outputFolder, 'hippocampal_order_parameter_R_5x2.svg');

print(f, pngName, '-dpng', '-r300');
print(f, svgName, '-dsvg');
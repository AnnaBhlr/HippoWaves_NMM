% Add necessary paths
repoRoot = fileparts(fileparts(mfilename('fullpath')));  % repo root (data/ paths)
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/brainwaves-master'));
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/neural-flows-master'));
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/ScientificColormaps'));
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/Hippocampus'));
addpath(genpath('/Users/abehler/Documents/Hippocampus/meshes/'));
%%
inputFolder = '/Volumes/NEWYSNG/Anna/HippocampalCorticalModeling/JR_Hippocampus_Cortex_naive_1to1';
outputFolder = '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/HippoCortex_1to1_couplingt2';
  
% theta = 'fast';
% simulations = {...
%     '20240703_110545', ... % delta p = -10
%     '20240703_140446'... % delta p = 10
%     };

theta = 'slow';
simulations = {...
    '20240704_050048', ... % delta p = -10
    '20240704_080008'... % delta p = 10
    };

inputFolderPaths = cellfun(@(s) fullfile(inputFolder, s), simulations, 'UniformOutput', false);

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

tSwitchOn = 2;
tSwitchOff = 10;
% Define colormap
colormap_main = readmatrix(fullfile(repoRoot, 'data', 'colormaps', 'RdYlBu_colormap.csv'));
colormap_main = flipud(colormap_main); % Flip to match the desired color scheme

%

fieldact_mesh_all = cell(1, length(inputFolderPaths));
fieldact_sphere_all = cell(1, length(inputFolderPaths));
time_all = cell(1, length(inputFolderPaths));

for datasetIdx = 1:length(inputFolderPaths)
    inputFolderPath = inputFolderPaths{datasetIdx}
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
        fieldact_mesh_all{datasetIdx} = fieldact(discardIdx:end, 1:hippoNnodes);
        fieldact_sphere_all{datasetIdx} = fieldact(discardIdx:end, hippoNnodes+1:end);
        time_all{datasetIdx} = time;
    end
end
%%
for datasetIdx = 1:length(inputFolderPaths)

    fieldact_mesh = fieldact_mesh_all{datasetIdx};
    fieldact_sphere = fieldact_sphere_all{datasetIdx};
    time = time_all{datasetIdx};
    % Video setup
    videoFilename = sprintf('%s_activity.mp4', simulations{datasetIdx});
    videoPath = fullfile(outputFolder, videoFilename);
    video = VideoWriter(videoPath, 'MPEG-4');
    video.FrameRate = 13;
    open(video);

    % Set color limits
    clims = [min(fieldact_mesh(:)), max(fieldact_mesh(:))];

    figure;
    set(gcf, 'Position', [100, 100, 900, 500]);

    for i = 1:length(time)
        clf;

        dataVid1 = fieldact_mesh(i, :);
        dataVid2 = fieldact_sphere(i, :);

        ax1 = subplot(1, 2, 1);
        figure_surf2(hippoFaces, hippoVertices, dataVid1, colormap_main, az, el);
        hold on;
        set(ax1, 'Position', [0.1, 0.05, 0.2, 0.8]);
        patch('faces', hippoFaces, 'vertices', hippoVertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
        clim(clims);

        ax2 = subplot(1, 2, 2);
        figure_surf2(hemiFaces, hemiVertices, dataVid2, colormap_main, az, el);
        hold on;
        set(ax2, 'Position', [0.4, 0.05, 0.53, 0.8]);
        patch('faces', hemiFaces, 'vertices', hemiVertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
        clim(clims);

        sgtitle(['Pyramidal post synaptic potential at ', sprintf('%.2f', time(i)), ' s']);

        % A-P annotation
        annotation('line', [0.1, 0.15], [0.15, 0.15], 'LineWidth', 2);
        annotation('textbox', [0.075, 0.13, 0.03, 0.03], 'String', 'A', 'EdgeColor', 'none', 'FontSize', 16);
        annotation('textbox', [0.145, 0.13, 0.03, 0.03], 'String', 'P', 'EdgeColor', 'none', 'FontSize', 16);

        % Optional coupling annotation (comment out if not needed)
        if time(i) >= tSwitchOn - relaxTime && time(i) <= tSwitchOff - relaxTime
            textBoxPosition = [0.225, 0.45, 0.3, 0.1];
            annotation('textbox', textBoxPosition, 'String', 'coupling', ...
                'HorizontalAlignment', 'center', 'FontSize', 16, 'EdgeColor', 'none');
            startPoint = [0.325, textBoxPosition(2)];
            endPoint = [0.425, textBoxPosition(2)];
            annotation('arrow', [startPoint(1), endPoint(1)], [startPoint(2), startPoint(2)], ...
                'HeadStyle', 'vback2', 'HeadWidth', 10, 'HeadLength', 10);
        end

        % Save frame to video
        print('-dpng', '-r150', 'temp_frame.png');
        img = imread('temp_frame.png');
        frame = im2frame(img);
        writeVideo(video, frame);
        delete('temp_frame.png');
    end

    close(video);
    close;
end

%%
% Define specific time points you want to capture (actual time values in seconds)
%timePoints = [0.73, 1, 1.27, 3.04, 3.41, 3.91, 4.2, 4.6,6.0, 7.5,8.0, 8.1,8.5];
timePoints = [2.9, 4.6, 5.5, 7.5, 8.5, 9.66];
% Initialize storage for order parameter data
all_RHippo = cell(1, length(inputFolderPaths));

for datasetIdx = 1:length(inputFolderPaths)

    fieldact_mesh = fieldact_mesh_all{datasetIdx};
    fieldact_sphere = fieldact_sphere_all{datasetIdx};
    time = time_all{datasetIdx};
    % Compute phase and order parameter
    yphase = unwrap(angle(hilbert(bsxfun(@minus,fieldact_sphere,mean(fieldact_sphere)))));
    RHippo = abs(mean(exp(1i * yphase), 2)); % Coherence 'order parameter'

    % Store R values
    all_RHippo{datasetIdx} = RHippo;

    % Find the indices corresponding to the specified time values
    indices = arrayfun(@(t) find(abs(time - t) == min(abs(time - t)), 1), timePoints);

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
        % clim([min(fieldact_mesh(:)), max(fieldact_mesh(:))]);
        % hippoFilename = fullfile(outputFolder, sprintf('hippocampus_%s_%.2f.png', simulations{datasetIdx}, time(i)));
        % print('-dpng', '-r300', hippoFilename);
        % close;

        % Plot and save hemisphere mesh
        figure;
        set(gcf, 'Position', [100, 100, 900, 500]);
        figure_surf2(hemiFaces, hemiVertices, dataVidHemi, colormap_main, az, el);
        %patch('faces', hemiFaces, 'vertices', hemiVertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
        title(sprintf('Dataset %s at t = %.2f s', simulations{datasetIdx}, time(i)));
        colorbar;
        clim([min(min(fieldact_sphere(200:end, :))), max(max(fieldact_sphere(200:end, :)))]);
        hemiFilename = fullfile(outputFolder, sprintf('hemisphere_%s_%.2f.png', simulations{datasetIdx}, time(i)));
        %material dull;
        %camlight;
        print('-dpng', '-r300', hemiFilename);
        close;
    end
end
%

f = figure;
set(gcf, 'Position', [100, 100, 900, 400]); % Set figure size
hold on;
for datasetIdx = 1:length(inputFolderPaths)
    if strcmp(theta, 'slow')
        if datasetIdx == 1
            c = '#76221F';
        elseif datasetIdx == 2
            c = '#80A5C0';
        end
    elseif strcmp(theta, 'fast')
        if datasetIdx == 1
            c = '#80A5C0';
        elseif datasetIdx == 2
            c = '#76221F';
        end
    end
    plot(time_all{datasetIdx}, smoothdata(all_RHippo{datasetIdx}, "gaussian", 30), ...
        'LineWidth', 1.5,'Color', c);
end
xlabel('Time (s)');
ylabel('Coherence (R)');
xlim([min(cellfun(@min, time_all)), max(cellfun(@max, time_all))]);
ylim([0, 1.05]); % R ranges from 0 to 1
box on; 
R_filename = fullfile(outputFolder, sprintf('order_parameter_R_%s.png', theta));
print('-dpng', '-r300', R_filename);
svgName = fullfile(outputFolder, sprintf('order_parameter_R_%s.svg', theta));
print(f, svgName, '-dsvg');
close; % Close the figure to free up memory

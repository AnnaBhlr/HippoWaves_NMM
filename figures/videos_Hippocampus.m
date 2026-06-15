repoRoot = fileparts(fileparts(mfilename('fullpath')));  % repo root (data/ paths)
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/brainwaves-master'))
addpath(genpath('/Users/abehler/Documents/MATLAB/cnem_25-05-23' ))
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/neural-flows-master'))
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/ScientificColormaps'))
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/Hippocampus/meshes'))

% get meshes
hippocampus;
hippoVertices=msh.POS; % in mm
hippoFaces=msh.TRIANGLES(:,1:3);
hippoNnodes = size(hippoVertices,1);

clear msh
%
relaxTime = 0.05; % relaxtion time, in s
% azimuth and elevation for plots ([0,90]=axial plane)
az = -130;
el = 15;

fontSize = 14;
%%
files = [...
    "20240513_191947", ...% slow theta, delta p = 0
    "20240513_193443", ... % slow theta, delta p = +5
    "20240513_184954" % slow theta, delta p = -10
    ];

% Loop over each folder/timepoint
for fIdx = 1:length(files)
    tag = files(fIdx);

    % Build paths dynamically
    inputDataFieldact = sprintf('/Volumes/NEWYSNG/Anna/HippocampalCorticalModeling/JR_FreesurferHippocampus/pGradient_exp0p3/%s/fieldact_%s.csv', tag, tag);
    inputDataTime = sprintf('/Volumes/NEWYSNG/Anna/HippocampalCorticalModeling/JR_FreesurferHippocampus/pGradient_exp0p3/%s/time_vector_%s.csv', tag, tag);

    % Define output video file with timestamp
    videoFilename = sprintf('/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/SupplementaryMaterial/HippoVid_%s.mp4', tag);

    % Load data
    tic
    pyrm = readmatrix(inputDataFieldact);
    time = readmatrix(inputDataTime);
    toc

    % Discard relaxation time
    dt = mean(diff(time));
    discardIdx = find(time >= relaxTime, 1, 'first');
    if ~isempty(discardIdx)
        time = time(discardIdx:end) - time(discardIdx);
        fieldact_mesh = pyrm(discardIdx:end, :);
    end

    % Create video writer
    video = VideoWriter(videoFilename, 'MPEG-4');
    video.FrameRate = 13;
    open(video);

    figure;
    set(gcf, 'Position', [100, 100, 900, 500]);
    clims = [min(fieldact_mesh(:)), max(fieldact_mesh(:))];
    colormap_main = flipud(readmatrix(fullfile(repoRoot, 'data', 'colormaps', 'RdYlBu_colormap.csv')));

    for i = 1:length(time)
        dataVid1 = fieldact_mesh(i, :);
        clf;

        figure_surf2(hippoFaces, hippoVertices, dataVid1, colormap_main, az, el);
        hold on;
        patch('faces', hippoFaces, 'vertices', hippoVertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
        title(sprintf('Local Field Potential, t = %.0f ms', time(i) * 1e3), 'FontSize', fontSize);
        clim(clims);
        % Add colorbar
        cb = colorbar;
        cb.Ticks = clims;
        cb.TickLabels = {'min', 'max'};
        cb.Position(4) = cb.Position(4)*0.5 ;  % shrink width by 50
        cb.FontSize = fontSize;  % adjust as needed
        cb.Position(2) = cb.Position(2) + 0.2;  % shift it up slightly if needed
        % Mini coordinate system
        axisLength = 10;
        x0 = -10; y0 = -10; z0 = -10;
        line([x0 x0+axisLength], [y0 y0], [z0 z0], 'Color', [0.5 0.5 0.5], 'LineWidth', 2); % X
        line([x0 x0], [y0 y0+axisLength], [z0 z0], 'Color', [0.5 0.5 0.5], 'LineWidth', 2); % Y
        line([x0 x0], [y0 y0], [z0 z0+axisLength], 'Color', [0.5 0.5 0.5], 'LineWidth', 2); % Z
        text(x0 + axisLength, y0, z0+2, 'M', 'FontSize', fontSize, 'Color', [0.5 0.5 0.5]);
        text(x0+2.5, y0, z0+2, 'L', 'FontSize', fontSize, 'Color', [0.5 0.5 0.5]);
        text(x0, y0 + axisLength, z0-2, 'A', 'FontSize', fontSize, 'Color', [0.5 0.5 0.5]);
        text(x0, y0, z0-2, 'P', 'FontSize', fontSize, 'Color', [0.5 0.5 0.5]);
        text(x0-2, y0, z0 + axisLength, 'S', 'FontSize', fontSize, 'Color', [0.5 0.5 0.5]);
        text(x0-2, y0, z0+2, 'I', 'FontSize', fontSize, 'Color', [0.5 0.5 0.5]);

        print('-dpng', '-r150', 'temp_frame.png');
        frame = im2frame(imread('temp_frame.png'));
        writeVideo(video, frame);
        delete('temp_frame.png');
    end

    close(video);
end

%% thumbnail
thumbnailFilename = '/Users/abehler/Library/Mobile Documents/com~apple~CloudDocs/Poster/2024_OHBM/HippoTHumbnail.png';
figure;
set(gcf, 'Position', [100, 100, 900, 500]);
clims = [min(fieldact_mesh(:)), max(fieldact_mesh(:))];
colormap_main = readmatrix(fullfile(repoRoot, 'data', 'colormaps', 'RdYlBu_colormap.csv'));
colormap_main = flipud(colormap_main); % Red = high, blue = low

% Extract the data for this time point
dataVid1 = fieldact_mesh(300, :);
clf;

figure_surf2(hippoFaces, hippoVertices, dataVid1, colormap_main, az, el);
hold on;
patch('faces', hippoFaces, 'vertices',hippoVertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
title(sprintf('Excitatory activity rate, t = %.2d ms', i));

clim(clims);

% mini coordinate system
axisLength = 10; % Adjust as needed
x0 = -10;
y0 = -10;
z0 = -10;
line([x0 x0+axisLength], [y0 y0], [z0 z0], 'Color', [0.5 0.5 0.5], 'LineWidth', 2); % X
line([x0 x0], [y0 y0+axisLength], [z0 z0], 'Color', [0.5 0.5 0.5], 'LineWidth', 2); % Y
line([x0 x0], [y0 y0], [z0 z0+axisLength], 'Color', [0.5 0.5 0.5], 'LineWidth', 2); % Z
text(x0 + axisLength, y0, z0+2, 'M', 'FontSize', 12, 'Color', [0.5 0.5 0.5]);
text(x0+2.5, y0, z0+2, 'L', 'FontSize', 12, 'Color', [0.5 0.5 0.5]);
text(x0, y0 + axisLength, z0-2, 'A', 'FontSize', 12, 'Color', [0.5 0.5 0.5]);
text(x0, y0, z0-2, 'P', 'FontSize', 12, 'Color', [0.5 0.5 0.5]);
text(x0-2, y0, z0 + axisLength, 'S', 'FontSize', 12, 'Color', [0.5 0.5 0.5]);
text(x0-2, y0, z0+2, 'I', 'FontSize', 12, 'Color', [0.5 0.5 0.5]);

% Save the current frame as a high-resolution PNG
print('-dpng', '-r300', thumbnailFilename);

%%

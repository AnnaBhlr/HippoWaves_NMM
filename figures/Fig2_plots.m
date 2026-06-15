repoRoot = fileparts(fileparts(mfilename('fullpath')));  % repo root (data/ paths)
≠%addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/brainwaves-master'))
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/neural-flows-master'))
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/ScientificColormaps'))
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/slanCM'))
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/Hippocampus'))
%%
% get meshes
hippocampus;
hippoVertices=msh.POS; % in mm
hippoFaces=msh.TRIANGLES(:,1:3);
clear msh

az = -130;
el = 15;

%% plain mesh (Fig. 1A)

patch('faces', hippoFaces, 'vertices', hippoVertices, 'facecolor',[1 1 1] , 'edgecolor', [0.5 0.5 0.5]);
view(az, el);
axis equal; axis off; 
outputFilename = sprintf('/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/Hippomesh.png');
%print('-dpng', '-r300', outputFilename);

%% Time series data for wave and synchronous state (Fig. 2)

%    
% "20240513_195002", ... % slow theta, delta p = 10
files = [...
    "20240513_191947", ...% slow theta, delta p = 0
    "20240513_193443", ... % slow theta, delta p = +5
    "20240513_184954" % slow theta, delta p = -10
    ];

% settings
colormap_main = readmatrix(fullfile(repoRoot, 'data', 'colormaps', 'RdYlBu_colormap.csv'));
colormap_main = flipud(colormap_main); % Red = high, blue = low

% Define specific time points you want to capture (actual time values in seconds)
timePoints = 6.2:.025:6.4;

index_1 = find(all(abs(hippoVertices - [-27, -11.5, -22]) < 1e-10, 2));
index_2 = find(all(abs(hippoVertices - [-30 -21 -11.5]) < 1e-10, 2));
index_3 = find(all(abs(hippoVertices - [-26 -34 -6.5]) < 1e-10, 2));
index_4 = find(all(abs(hippoVertices - [-19,-37.5000000000000,1])< 1e-10, 2));

color_dark = '#4C5762';
color_light = '#8B9A8B';
% Define colors for each line (customize as needed)
% custom_colors = [
%     184/255, 134/255, 11/255;   % #b8860b
%     2/255, 147/255, 134/255     % #029386
% ];

custom_colors = [
    0.2, 0.2, 0.2;    % #333333
    0.5, 0.5, 0.5;  % #808080
    0.8, 0.8, 0.8];   % #CCCCCC

numFiles = size(files, 2);

for k = 1:numFiles
    % Extract the current filename (trim whitespace)
    file = files(k);

    [RHippo, time] = plot_snapshots_timeseries(k, file, timePoints, index_1, index_2,index_3, index_4, hippoVertices, hippoFaces, ...
        colormap_main, az, el, color_dark, color_light, custom_colors);

        % Store in cells
    RHippo_cell{k} = RHippo;
    time_cell{k} = time;
    close all;
end

% Create combined coherence plot
f = figure;
hold on;


for k = 1:numFiles
    % Smooth the RHippo data
    RHippo_smoothed = smoothdata(RHippo_cell{k}, "gaussian", 30);
    
    % Plot against actual time
    plot(time_cell{k}, RHippo_smoothed, ...
        'LineWidth', 1.5, ...
        'Color',  custom_colors(k, :)); 
end

hold off;

% Labeling and aesthetics
xlabel('Time / s');
ylabel('Global coherence');
ylim([0.5 1.05]);
set(gcf, 'Units', 'centimeters', 'Position', [10, 10, 28.2, 7.5]);
set(gca, 'FontSize', 12);
box on

% Save the combined plot
outputFilename = '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/Combined_coherence.png';
print('-dpng', '-r300', outputFilename);
svgName = '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/Combined_coherence.svg';
print(f, svgName, '-dsvg');
close all

function [RHippo, time] = plot_snapshots_timeseries(k, file, timePoints, index_1,index_2, index_3, index_4, hippoVertices, hippoFaces, ...
    colormap_main, az, el, color_dark, color_light, colors, time)

    disp('file gets processed')
  
    inputDataFieldact = sprintf( ...
        '/Volumes/NEWYSNG/Anna/HippocampalCorticalModeling/JR_FreesurferHippocampus/pGradient_exp0p3/%s/fieldact_%s.csv', ...
        file, file ...
        );
    inputDataTime = sprintf( ...
        '/Volumes/NEWYSNG/Anna/HippocampalCorticalModeling/JR_FreesurferHippocampus/pGradient_exp0p3/%s/time_vector_%s.csv', ...
        file, file ...
        );
    
    % Load the data
    tic
    pyrm = readmatrix(inputDataFieldact);
    time = readmatrix(inputDataTime);
    toc

    relaxTime = 0.05; % Relaxation time, in seconds
    discardIdx = find(time >= relaxTime, 1, 'first')
    if ~isempty(discardIdx)
        time = time(discardIdx:end) - time(discardIdx);
        fieldact_mesh = pyrm(discardIdx:end, :);
    end
    
    % hilbert transform of field activity 
    yphaseHippo=unwrap(angle(hilbert(bsxfun(@minus,fieldact_mesh,mean(fieldact_mesh)))));
    % order parameter
    RHippo = abs(mean(exp(1i*yphaseHippo),2)); % coherence 'order parameter'

    % Find the indices corresponding to these time values
    indices = arrayfun(@(t) find(abs(time - t) == min(abs(time - t)), 1), timePoints);
    clims = [min(min(fieldact_mesh(300:end, :))), max(max(fieldact_mesh(300:end, :)))]
    
    figure;
    % Loop through only the specified time points
    for idx = 1:length(indices)
        i = indices(idx); % Get the index corresponding to the specific time value

        % Extract the data for this time point
        dataVid1 = fieldact_mesh(i, :);
        
        clf;

        % Plot the mesh
        figure_surf2(hippoFaces, hippoVertices, dataVid1, colormap_main, az, el);
        hold on;
        %patch('faces', hippoFaces, 'vertices', hippoVertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
        title(sprintf('Excitatory activity rate, t = %.2d s', timePoints(idx)));

        % Set color limits and add colorbar
        clim(clims);
        colormap(colormap_main);
        colorbar; 

        % Add markers for anterior/posterior indices (ensure they are plotted last)
        [sphereX, sphereY, sphereZ] = sphere(20); % 20 faces for smoothness
        sphereRadius = 1; % Adjust size as needed

        % % Shift sphere to anterior point
        % x = hippoVertices(index_1,1) + sphereX * sphereRadius;
        % y = hippoVertices(index_1,2) + sphereY * sphereRadius;
        % z = hippoVertices(index_1,3) + sphereZ * sphereRadius;
        % patch(x, y, z, 'r', 'FaceAlpha', 1, 'FaceColor', 'k', 'EdgeColor', 'None', 'LineWidth', 1);
        % % Shift sphere to anterior point
        % x = hippoVertices(index_2,1) + sphereX * sphereRadius;
        % y = hippoVertices(index_2,2) + sphereY * sphereRadius;
        % z = hippoVertices(index_2,3) + sphereZ * sphereRadius;
        % patch(x, y, z, 'r', 'FaceAlpha', 1, 'FaceColor', 'k', 'EdgeColor', 'None', 'LineWidth', 1);
        % % Shift sphere to anterior point
        % x = hippoVertices(index_3,1) + sphereX * sphereRadius;
        % y = hippoVertices(index_3,2) + sphereY * sphereRadius;
        % z = hippoVertices(index_3,3) + sphereZ * sphereRadius;
        % patch(x, y, z, 'r', 'FaceAlpha', 1, 'FaceColor', 'k', 'EdgeColor', 'None', 'LineWidth', 1);
        % 
        % % Shift sphere to anterior point
        % x = hippoVertices(index_4,1) + sphereX * sphereRadius;
        % y = hippoVertices(index_4,2) + sphereY * sphereRadius;
        % z = hippoVertices(index_4,3) + sphereZ * sphereRadius;
        % patch(x, y, z, 'r', 'FaceAlpha', 1, 'FaceColor', 'k', 'EdgeColor', 'None', 'LineWidth', 1);

        % Save the current frame as a high-resolution PNG
        outputFilename = sprintf('/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/Hippocampus_pGradient/%s_timepoint_%d.png', file, i);
        print('-dpng', '-r300', outputFilename);
    end
    close all;

    % Function to convert hex to RGB (0–1 range)
    hex2rgb = @(hex) sscanf(hex(2:end), '%2x%2x%2x', [1 3]) / 255;
    
    % Define your hex colours
    color1 = hex2rgb('#102820');
    color2 = hex2rgb('#8a9a5b');
    color3 = hex2rgb('#caba9c');
    color4 = hex2rgb('#8a6240');

    % set time for plot
    t_start = 6;
    t_end = 7;
    t_start_idx = find(time >= t_start, 1, 'first')
    t_end_idx = find(time <= t_end, 1, 'last')

    f = figure;
    plot(time(t_start_idx:t_end_idx),fieldact_mesh(t_start_idx:t_end_idx, index_1), ...
        'LineWidth', 1.5, 'Color', color1);
    hold on;
    plot(time(t_start_idx:t_end_idx),fieldact_mesh(t_start_idx:t_end_idx, index_2), ...
        'LineWidth', 1.5, 'Color', color2);
    plot(time(t_start_idx:t_end_idx),fieldact_mesh(t_start_idx:t_end_idx, index_3), ...
        'LineWidth', 1.5, 'Color', color3);
    plot(time(t_start_idx:t_end_idx),fieldact_mesh(t_start_idx:t_end_idx, index_4), ...
        'LineWidth', 1.5, 'Color', color4);

    xlim([time(t_start_idx) time(t_end_idx)]);
        % Add vertical lines at the time points where the mesh is plotted
    for idx = 1:length(indices)
        i = indices(idx)- 400; 
        %xline(i, '-', 'Color', colors(k, :), 'LineWidth', 0.8);
    end

    % plot(fieldact_mesh(201:600,index_ant), ...
    %     'LineWidth', 1.5,'Color',color_dark);
    % hold on;
    % plot(fieldact_mesh(201:600,index_pos), ...
    %     'LineWidth', 1.5,'Color',color_light, 'LineStyle', '--');
    % xlim([1 400]);

    % Add labels and legend
    xlabel('Time / s');
    ylabel('Activity');
    
    % Set figure properties for high-resolution export
    % Set figure size to 13 cm wide and 2 cm high
    set(gcf, 'Units', 'centimeters', 'Position', [10, 10, 28.2, 5.9]);
    set(gca, 'FontSize', 12); % Larger font size
    
    % Save as PNG with 300 DPI
    fileName = sprintf(...
        '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/%s_activity_comparison.png', ...
        file);
    print('-dpng', '-r300', fileName);
    svgName = sprintf(...
    '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/%s_activity_comparison.svg', ...
    file);
    print(f, svgName, '-dsvg');
 
    % figure;
    % plot(smoothdata(RHippo,"gaussian",30), ...
    %     'LineWidth', 1.5, 'Color', color_dark);
    % xlabel('Time / s');
    % ylabel('Global coherence');
    % ylim([0.5 1])
    % % Set figure properties for high-resolution export
    % set(gcf, 'Color', 'w', 'Position', [100 100 600 600]); % White background, 800x600 pixels
    % set(gca, 'FontSize', 12); % Larger font size
    % 
    %     % Save as PNG with 300 DPI
    % fileName = sprintf(...
    %     '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/%s_coherence.png', ...
    %     file);
    % print('-dpng', '-r300', fileName);


end

%% uniform p over hippocampus and cortex surface

% Create the vector p with all entries equal to 2
p = 2 * ones(3718, 1);
colorscheme = 'YlOrBr';
% Create the figure
figure;
my_colormap = slanCM(colorscheme);
%my_colormap = flipud(my_colormap); % Uncomment to invert the colormap

% Plot the surface with the specified colormap and viewing angle
figure_surf2(hippoFaces, hippoVertices, p, my_colormap, -130, 15);

% Hold on to add additional elements to the plot
hold on;

% Add a patch to display the edges of the surface
patch('faces', hippoFaces, 'vertices', hippoVertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
clim([2, 2.1]);
fileName = '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/p_uniform.png';
%print('-dpng', '-r300', fileName);

p = 2 * ones(9204, 1);
figure;
my_colormap = slanCM(colorscheme);
%my_colormap = flipud(my_colormap); % Uncomment to invert the colormap

% Plot the surface with the specified colormap and viewing angle
figure_surf2(hemiFaces, hemiVertices, p, my_colormap, -130, 15);

% Hold on to add additional elements to the plot
hold on;

% Add a patch to display the edges of the surface
patch('faces', hemiFaces, 'vertices', hemiVertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
clim([2, 2.1]);
fileName = '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/p_uniform_cortex.png';
print('-dpng', '-r300', fileName);

%%

% Create the vector p1 ranging from 2 to 1.05 times 2
p1 = linspace(2*1.1, 2, 3718)';

% Create the vector p2 ranging from 2 to 0.9 times 2
p2 = linspace(2*0.9, 2, 3718)';

colorscheme = 'YlOrBr';
% Create the figure
figure;
my_colormap = slanCM(colorscheme);
%my_colormap = flipud(my_colormap); % Uncomment to invert the colormap

% Plot the surface for p1
figure_surf2(hippoFaces, hippoVertices, p1, my_colormap, -130, 15);

% Add a patch to display the edges of the surface
patch('faces', hippoFaces, 'vertices', hippoVertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);

% Set the color limits to include the range of p1 and p2
clim([2 2.2]);
colorbar;

% Save the figure
fileName = '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/p_heavytail_10.png';
print('-dpng', '-r300', fileName);
%% vector fields for delta p =0 and delta p = 5


inputFolderPath = '/Volumes/DATA/Hippocampus';
subFolders = {"20240513_193443", ... % slow theta, delta p = +5
              "20240513_191947" ...% slow theta, delta p = 0
             };

timesToPlot = [1, 3, 6];


for k = 1:length(subFolders)

   inputDataTime = sprintf( ...
        '/Volumes/DATA/Hippocampus/%s/time_vector_%s.csv', ...
        subFolders{k}, subFolders{k} ...
        );

    time = readmatrix(inputDataTime);

    relaxTime = 0.05; % Relaxation time, in seconds
    discardIdx = find(time >= relaxTime, 1, 'first');
    if ~isempty(discardIdx)
        time = time(discardIdx:end) - time(discardIdx);
    end
    thisFolder = fullfile(inputFolderPath, subFolders{k});
    thisFile = sprintf('%s_v.mat',subFolders{k} );
    thisFilePath = fullfile(thisFolder, thisFile);

    load(thisFilePath);

    for i = 1:length(timesToPlot)
        idxTime = find(time >= timesToPlot(i), 1);

        % vector field
        f = figure;
        patch('faces', hippoFaces, 'vertices',hippoVertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
        hold on;
        plot_vectors_on_mesh(hippoFaces, hippoVertices, vHippo, idxTime, -130, 15,[4 4]);

        % Construct the filename to save the figure
        figName = sprintf('%s_VectorField_Time_%.2f_s.png', subFolders{k},double(timesToPlot(i)));
        savePath = fullfile('/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems', figName);

        % Save the figure with 300 DPI
        print(f, savePath, '-dpng', '-r300');
        close(f);

    end
end

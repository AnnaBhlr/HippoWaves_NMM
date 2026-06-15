
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/ScientificColormaps'))
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/slanCM'))
%%
% read the data for the plots from the files
inputFolderPath = '/Volumes/NEWYSNG/Anna/HippocampalCorticalModeling/JR_FreesurferHippocampus/pGradient_exp0p3';

theta = 'fast';

% Read the data from the csv file
all_R = readmatrix(fullfile(inputFolderPath, sprintf('%s_mean_R_overT.csv', theta)));
% Get the numeric headers (column names)
numericHeaders = all_R(1,:);
% Convert the numeric headers to strings
stringHeaders = arrayfun(@num2str, numericHeaders, 'UniformOutput', false);
% Create a new table with the string headers
all_R = array2table(all_R(2:end,:), 'VariableNames', stringHeaders);

% Read the data from the csv file
all_R_smooth = readmatrix(fullfile(inputFolderPath, sprintf('%s_mean_R_smooth_overT.csv', theta)));
% Get the numeric headers (column names)
numericHeaders = all_R_smooth(1,:);
% Convert the numeric headers to strings
stringHeaders = arrayfun(@num2str, numericHeaders, 'UniformOutput', false);
% Create a new table with the string headers
all_R_smooth = array2table(all_R_smooth(2:end,:), 'VariableNames', stringHeaders);

% plot the figures   
data_tables_R = {all_R, all_R_smooth};  
smooth_labels = {'notsmoothed', 'smoothed'};  % Labels corresponding to each table
labelNames = all_R.Properties.VariableNames;
numericLabels = str2double(labelNames);

for k = 1:length(data_tables_R)
    data_R = data_tables_R{k};  % Extract each table for processing
    smooth = smooth_labels{k};  % Get the label for the current table

    outputFigureR = fullfile(inputFolderPath, sprintf('R_%s_%s.png', smooth, theta));
    
    R_matrix = data_R{:,:};
    f = figure('Position', [10 10 900 300]);
    % Define the position for each subplot
    % [left bottom width height]
    position1 = [0.06, 0.2, 0.6, 0.7]; % Adjust these values as needed
    position2 = [0.67, 0.2, 0.2, 0.7]; % Adjust these values as needed
    colorbarPosition = [0.62, 0.2, 0.3, 0.7]; % Position for the colorbar
    fontName = 'Verdana';
    labelFontSize = 16;
    tickLabelSize = 14;
    % First subplot with 1:2 height to width ratio
    ax1 = subplot('Position', position1);
    %R_colormap = flipud(slanCM('jet'));
    %R_colormap = flipud(crameri('lapaz'));

    %colormap(R_colormap);
    imagesc(R_matrix.');
    clim([0.5, 1]); % Adjust the color scale
    yticks(1:length(labelNames));
    % Modify y-tick labels to display only every second label
    labels = labelNames;  % Get the current y-tick labels
    modified_labels = labels;  % Create a copy of the labels
    modified_labels(mod(1:length(labels), 2) == 0) = {''};  % Set every second label to an empty string
    yticklabels(modified_labels);  % Apply the modified labels
    set(gca,'YDir','normal', 'FontName',fontName);
    % Retrieve the default x-tick positions and calculate labels
    defaultXTicks = get(gca, 'XTick');  % Get the default x-tick positions
    timePerIndex = 0.01;  % Time per index in seconds (10ms)
    xtickLabels = arrayfun(@(x) sprintf('%d', x * timePerIndex), defaultXTicks, 'UniformOutput', false);
    set(gca, 'XTickLabels', xtickLabels,'FontSize', tickLabelSize, 'FontName',fontName);  % Set new x-tick labels
    set(ax1, 'TickLength', [0.01, 0.025]); % Set tick length
    
    xlabel('Time / s', 'FontSize', labelFontSize, 'FontName',fontName);
    ylabel('\Delta p / %', 'FontSize', labelFontSize, 'FontName',fontName);
    
    % Second subplot with 1:1 ratio, same height as the first
    ax2 = subplot('Position', position2);
    yData = numericLabels;
    xData = mean(R_matrix,1);
    plot(xData, yData, 'Color', 'black', 'LineWidth', 1.5); 
    xlabel('Velocity / ms^{-1}', 'FontSize',labelFontSize, 'FontName',fontName);
    xlim([0.5, 1]);
    ylim([-10.5, 10.5])
    set(ax2, 'YTick', linspace(min(yData), max(yData), 5)); % Add y-ticks
    % Set x-tick label size for the second plot
    set(ax2, 'FontSize', tickLabelSize, 'FontName',fontName);  % Adjust this to match the desired font size
    set(ax2, 'YTickLabel', {}); % Remove y-tick labels
    set(ax2, 'TickLength', [0.025, 0.25]); % Ensure tick length matches ax1
    % Set x-axis labels
    xlabel(ax1, 'Time / s', 'FontSize', labelFontSize, 'FontName',fontName);
    xlabel(ax2, 'R / a.u.', 'FontSize', labelFontSize, 'FontName',fontName);
    % Assuming xData contains the data for the x-axis
    minY = min(yData);
    maxY = max(yData);
    set(ax2, 'YTick', minY:1:maxY)
    
    pos1 = get(get(ax1, 'XLabel'), 'Position');
    set(get(ax1, 'XLabel'), 'Position', [pos1(1), -2.29, pos1(3)]);
    
    % Create a new axes for the colorbar
    cbaxes = axes('Position', colorbarPosition, 'Visible', 'off'); % Create new axes for colorbar
    cbar = colorbar(cbaxes, 'Visible', 'on'); % Create the colorbar
    %colormap(cbaxes,R_colormap); % Apply the same colormap as the first subplot
    clim(cbaxes, [0.5, 1]); % Match the color scaling
    set(cbar, 'FontSize', tickLabelSize);  % Increase colorbar tick label size
    ylabel(cbar, 'R / a.u.', 'FontSize', labelFontSize, 'FontName',fontName,'Rotation', 270);  % Set vertical title next to the colorbar
    % Adjust the position of the colorbar label to align with the y-axis label of the first plot
    cbar.Label.Position = [4.5, 0.5, 0];
    
    % Save the figure with 300 DPI
    print(f, outputFigureR, '-dpng', '-r300');
    close;
end

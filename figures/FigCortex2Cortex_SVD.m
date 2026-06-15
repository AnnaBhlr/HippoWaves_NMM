repoRoot = fileparts(fileparts(mfilename('fullpath')));  % repo root (data/ paths)
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/brainwaves-master'))
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/neural-flows-master'))
addpath(genpath('/Volumes/NEWYSNG/Anna/MATLAB/ScientificColormaps'))
%addpath(genpath('/Volumes/NEWYSNG/Anna/HippocampalCorticalModeling/JR_Hippocampus_Cortex_naive_1to1/'))
addpath(genpath('/Users/ab799/Documents/Hippocampus/meshes/'))
colormap= readmatrix(fullfile(repoRoot, 'data', 'colormaps', 'RdYlBu_colormap.csv'));
%%
fsaverage5_hemi_L_pial_noMedialWall;
hemiVertices = msh.POS; % in mm
hemiFaces = msh.TRIANGLES(:, 1:3);

% Define File Paths
data_path_1 = '/Volumes/NEWYSNG/Anna/HippocampalCorticalModeling/cortex2hippo/Output_Hippowaves_slow_neg10_010226/20260204_215244/FlowPotential_cortex.csv'; 
[parentFolder, ~, ~] = fileparts(data_path_1);   % .../20250919_140940
[~, name1]      = fileparts(parentFolder);  % '20250919_140940'

data_path_2 = '/Volumes/NEWYSNG/Anna/HippocampalCorticalModeling/cortex2hippo/Output_Hippowaves_slow_pos10_010226/20260204_162512/FlowPotential_cortex.csv';
[parentFolder, ~, ~] = fileparts(data_path_2);   % .../20250919_140940
[~, name2]      = fileparts(parentFolder);  % '20250919_140940'

save_dir = '/Users/ab799/Library/Mobile Documents/com~apple~CloudDocs/manuscript_HippoWaves/FigureItems/Cortex2Hippo';

% Load CSV Files (Skipping Header)
data1 = readmatrix(data_path_1, 'NumHeaderLines', 1);
data2 = readmatrix(data_path_2, 'NumHeaderLines', 1);

nKeep = 1495% + 1;   % 996 columns
data1 = data1(:, 1:nKeep);
data2 = data2(:, 1:nKeep);

% Run for Both Data Sets
process_svd_and_plot(data1, name1, save_dir, hemiFaces, hemiVertices);
process_svd_and_plot(data2, name2, save_dir, hemiFaces, hemiVertices);

% Function to Perform SVD and Plot
function process_svd_and_plot(data, dataset_name, save_dir, faces, vertices)

    [U, S, V] = svd(data, 'econ');
    singular_values = diag(S);
    total_variance = sum(singular_values.^2);
    variance_explained = (singular_values.^2) / total_variance * 100;

    disp(['Variance explained by first four singular values for ', dataset_name, ':']);
    disp(variance_explained(1:4));
    mini = min(U(:,1), [], 'all');
    maxi = max (U(:,1), [], 'all');
    if mini >= maxi
        clims = [-mini, mini];
    else 
        clims = [-maxi, maxi];
    end

    % Plot and Save First Four Left Singular Vectors
    for i = 1:3
        az = -130;
        el = 15;
        figure;
        figure_surf2(faces, vertices, -U(:, i), colormap, az, el);
        hold on;
        patch('faces', faces, 'vertices',vertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
        clim(clims);
        colorbar;
        title(['Left Singular Vector ' num2str(i) ' (' dataset_name ')']);
        saveas(gcf, fullfile(save_dir, sprintf('%s_Left_Singular_Vector_cortex_%d.png', dataset_name, i))); % Save figure
        close(gcf); % Close figure to avoid clutter
    end

    for i = 1:3
        az = 100;
        el = 0;
        figure;
        figure_surf2(faces, vertices, -U(:, i), colormap, az, el);
        hold on;
        patch('faces', faces, 'vertices',vertices, 'facecolor', 'none', 'edgecolor', [0.5 0.5 0.5]);
        clim(clims);
        colorbar;
        title(['Left Singular Vector ' num2str(i) ' (' dataset_name ')']);
        saveas(gcf, fullfile(save_dir, sprintf('%s_Left_Singular_Vector_medial_cortex_%d.png', dataset_name, i))); % Save figure
        close(gcf); % Close figure to avoid clutter
    end
    % Plot Right Singular Vectors in a Single Plot
    f = figure;
    set(gcf, 'Position', [100, 100, 900, 400]);
    hold on;
    time = (0:size(V, 1)-1) * 1e-3; % Convert to seconds

    plot(time, -smoothdata(V(:, 1)* singular_values(1), 'gaussian', 30), 'LineWidth', 1.5);
    plot(time, -smoothdata(V(:, 2)* singular_values(2), 'gaussian', 30),'LineWidth', 1.5);

    %plot(time, smoothdata(V(:, 2)* singular_values(2), 'gaussian', 30),'LineWidth', 1.5);
    %plot(time, smoothdata(V(:, 3)* singular_values(3), 'gaussian', 30),'LineWidth', 1.5);
    %plot(time, smoothdata(V(:, 4), 'gaussian', 30));
    hold off;
    xlabel('Time (s)');
    ylabel('Amplitude');
    xticks(0:.1:1.5);
    %ylim([-0.1 0.1])
    % Show all axes
    ax = gca;
    ax.XAxis.Visible = 'on';
    ax.YAxis.Visible = 'on';
    ax.Box = 'on';

    saveas(gcf, fullfile(save_dir, sprintf('%s_Right_Singular_Vectors_1and2_hemi.png', dataset_name))); % Save figure
    svgName = fullfile(save_dir, sprintf('%s_Right_Singular_Vectors_1and2_hemi.svg', dataset_name));
    print(f, svgName, '-dsvg');
    close(gcf); % Close figure to avoid clutter
end


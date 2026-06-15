% analyze_single_simulation.m
% ---------------------------------------------------------------------------
% Full analysis of ONE hippocampus-only simulation and ONE coupled simulation,
% from reading the raw output to a single summary figure.
%
% For each run it:
%   1. reads the time vector and field activity (output/<timestamp>/...),
%   2. discards the relaxation transient,
%   3. computes the phase coherence R (order parameter)        -> compute_coherence.m
%   4. computes the traveling-wave velocity vector field        -> compute_velocity_field.m
%      and from it the mean phase SPEED and mean velocity ANGLE over time,
%   5. plots R, mean speed and mean angle for both runs in ONE figure.
%
% All three quantities reproduce the manuscript calculations:
%   R           = |<exp(i*phase)>_nodes|                  (getVelocityHippocampus.m)
%   mean speed  = mean(v.vnormp, 2)                       (getVelocityHippocampus.m)
%   mean angle  = <acos(v_y/|v|)>_nodes  (angle to +Y / A-P axis, radians 0..pi)
%                                                         (meanVelocityAngle.m)
%
% For a COUPLED run only the hippocampus block (first hippoNnodes columns of the
% field activity) is analysed, matching figures/FigCortex2Hippo_VecFieldsHippo.m.
%
% Requires external MATLAB toolboxes (see README -> Dependencies -> MATLAB):
%   neural-flows, brainwaves (mesh_laplacian_uniform), cnem (phaseflow_cnem).
% ---------------------------------------------------------------------------

clear; close all;

% --- repo root + internal paths ---
repoRoot = fileparts(fileparts(mfilename('fullpath')));   % .../HippoCortexWaves
addpath(fullfile(repoRoot, 'analysis'));                  % compute_coherence / _velocity_field / _velocity_angle
addpath(fullfile(repoRoot, 'data', 'meshes'));            % hippocampus.m mesh script

% --- external toolboxes: set these to YOUR local installs ---
% addpath(genpath('/path/to/neural-flows'));
% addpath(genpath('/path/to/brainwaves'));
% addpath(genpath('/path/to/cnem'));

% --- the two single runs to analyse ---
% Defaults point at the bundled example data (out/...); change to your own
% output/<timestamp> folders to analyse new runs.
simHippoDir   = fullfile(repoRoot, 'out', 'example_hippocampus', '20240513_174237');
simCoupledDir = fullfile(repoRoot, 'out', 'example_coupled',     '20260204_213749');

relaxTime  = 0.05;   % s, discarded transient
smoothWin  = 30;     % gaussian smoothing window for R / speed (as in the figures)

% --- hippocampus surface mesh (shared: a coupled run stores the hippocampus
%     block first, so its first hippoNnodes columns use this same mesh) ---
hippocampus;                              % defines struct 'msh'
hippoVertices = msh.POS;                  % N x 3, in mm
hippoFaces    = msh.TRIANGLES(:, 1:3);
hippoNnodes   = size(hippoVertices, 1);
clear msh

% --- analyse each run ---
H = analyse_run(simHippoDir,   hippoNnodes, hippoVertices, hippoFaces, relaxTime);
C = analyse_run(simCoupledDir, hippoNnodes, hippoVertices, hippoFaces, relaxTime);

% --- single summary figure ---
figure('Position', [100 100 900 800]);

subplot(3,1,1);
plot(H.time, smoothdata(H.R, "gaussian", smoothWin), 'LineWidth', 1.5); hold on;
plot(C.time, smoothdata(C.R, "gaussian", smoothWin), 'LineWidth', 1.5);
ylabel('R'); ylim([0 1]); title('Phase coherence (order parameter)');
legend('hippocampus', 'coupled', 'Location', 'best');

subplot(3,1,2);
plot(H.time, smoothdata(H.meanSpeed, "gaussian", smoothWin), 'LineWidth', 1.5); hold on;
plot(C.time, smoothdata(C.meanSpeed, "gaussian", smoothWin), 'LineWidth', 1.5);
ylabel('mean |v| (m s^{-1})'); title('Mean phase speed');

subplot(3,1,3);
plot(H.time, H.meanAngle, 'LineWidth', 1.5); hold on;
plot(C.time, C.meanAngle, 'LineWidth', 1.5);
ylabel('mean angle (rad)'); ylim([0 pi]); yticks([0 pi/2 pi]); yticklabels({'0','\pi/2','\pi'});
xlabel('time (s)'); title('Mean velocity angle to +Y (A-P) axis');

% optional: save next to the runs
% print(gcf, fullfile(repoRoot, 'output', 'R_speed_angle.png'), '-dpng', '-r300');


% =========================== local function ===============================
function out = analyse_run(simDir, hippoNnodes, vertices, faces, relaxTime)
    % timestamp = the run-folder name (used in the csv filenames)
    ts = regexp(simDir, '\d{8}_\d{6}', 'match', 'once');
    if isempty(ts), [~, ts] = fileparts(simDir); end

    time     = readmatrix(fullfile(simDir, sprintf('time_vector_%s.csv', ts)));
    fieldact = readmatrix(fullfile(simDir, sprintf('fieldact_%s.csv',   ts)));

    % keep the hippocampus block (coupled runs: hippocampus first, then cortex)
    fieldact = fieldact(:, 1:hippoNnodes);

    % discard the relaxation transient
    dt = mean(diff(time));
    k0 = find(time >= relaxTime, 1, 'first');
    if ~isempty(k0)
        time     = time(k0:end) - time(k0);
        fieldact = fieldact(k0:end, :);
    end

    % (1) coherence / order parameter
    out.R = compute_coherence(fieldact);

    % (2) traveling-wave velocity field
    v = compute_velocity_field(fieldact, vertices, faces, dt);

    % mean phase speed over time (as in getVelocityHippocampus.m)
    out.meanSpeed = mean(v.vnormp, 2);

    % mean velocity angle to the +Y (A-P) axis over time (as in meanVelocityAngle.m)
    out.meanAngle = compute_velocity_angle(v);

    out.time = time;
end

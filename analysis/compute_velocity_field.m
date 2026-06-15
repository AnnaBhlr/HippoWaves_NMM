function [v, yphase] = compute_velocity_field(fieldact, vertices, faces, dt, tau, nSteps)
%COMPUTE_VELOCITY_FIELD  Traveling-wave velocity vector field on a surface mesh.
%
%   [v, yphase] = COMPUTE_VELOCITY_FIELD(fieldact, vertices, faces, dt)
%   returns the full velocity vector field v of the traveling waves in the
%   field activity fieldact (T timepoints x N nodes) on the surface mesh given
%   by vertices (N x 3, in mm) and faces. dt is the time step in seconds.
%
%   Pipeline (extracted from figures/FigCortex2Hippo_VecFieldsHippo.m):
%     1. de-mean each node's time series;
%     2. analytic (Hilbert) signal per node;
%     3. implicit Laplacian smoothing of the analytic signal over the mesh
%        (nSteps steps of strength tau, solving (I + tau*L) * Ynew = Y);
%     4. unwrap the (unit-phasor) phase in time -> yphase;
%     5. estimate the velocity vector field with phaseflow_cnem (dt in ms).
%
%   Optional arguments:
%       tau     smoothing strength per step   (default 0.2, try 0.05-0.5)
%       nSteps  number of diffusion steps      (default 5,   try 1-10)
%
%   Returns:
%       v       velocity vector field (as returned by phaseflow_cnem)
%       yphase  smoothed, unwrapped phase (T x N) used to estimate v
%
%   Requires the external MATLAB toolboxes (see README -> Dependencies -> MATLAB):
%     - brainwaves / neural-flows  (mesh_laplacian_uniform)
%     - cnem                       (phaseflow_cnem)

    if nargin < 5 || isempty(tau),    tau = 0.2;  end
    if nargin < 6 || isempty(nSteps), nSteps = 5; end

    nNodes = size(vertices, 1);

    % --- implicit Laplacian smoothing operator ---
    L = mesh_laplacian_uniform(faces, nNodes);
    S = speye(nNodes) + tau * L;            % implicit diffusion step
    [Rchol, p] = chol(S, 'lower');          % factor once (works if S is SPD)
    useChol = (p == 0);

    % de-mean each node's time series
    fieldact0 = bsxfun(@minus, fieldact, mean(fieldact, 1));

    % analytic signal per node (Hilbert along time dimension)
    analytic = hilbert(fieldact0);          % T x N, complex

    % spatial smoothing in the node dimension (solve N x T system)
    Y = analytic.';                         % N x T
    for s = 1:nSteps
        if useChol
            Y = Rchol' \ (Rchol \ Y);       % solves (I + tau*L) * Ynew = Y
        else
            Y = S \ Y;
        end
    end
    analytic_smooth = Y.';                  % back to T x N

    % normalise to unit phasor, then unwrap phase in time per node
    phasor = analytic_smooth ./ max(abs(analytic_smooth), eps);
    yphase = unwrap(angle(phasor), [], 1);

    % velocity vector field (dt converted from s to ms)
    v = phaseflow_cnem(yphase, vertices, dt * 1000, 5);
end

function R = compute_coherence(fieldact)
%COMPUTE_COHERENCE  Kuramoto phase-coherence ("order parameter") over time.
%
%   R = COMPUTE_COHERENCE(fieldact) returns the global phase-coherence time
%   course R (T x 1) from field activity fieldact (T timepoints x N nodes).
%   For each node the instantaneous phase is obtained from the analytic
%   (Hilbert) signal of the de-meaned activity; R(t) is the modulus of the
%   mean phasor across nodes:  R = |<exp(i*phase)>_nodes|.
%
%   This is the coherence calculation that was inlined (identically) in the
%   manuscript figure scripts before plotting:
%       figures/Fig2_plots.m
%       figures/Fig4_SnapshotsAndR.m
%       figures/FigCortex2Hippo_SnapshotAndR.m
%       figures/Fig_Suppl_CouplingSweep.m
%   It has been extracted here so the calculation lives with the analysis code
%   and the figure scripts can simply plot its output.

    yphase = unwrap(angle(hilbert(bsxfun(@minus, fieldact, mean(fieldact)))));
    R = abs(mean(exp(1i * yphase), 2));   % coherence 'order parameter'
end

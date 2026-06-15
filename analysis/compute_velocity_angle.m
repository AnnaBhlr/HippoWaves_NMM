function theta = compute_velocity_angle(v, smoothWin)
%COMPUTE_VELOCITY_ANGLE  Mean velocity angle to the +Y (anterior-posterior) axis.
%
%   theta = COMPUTE_VELOCITY_ANGLE(v) returns the time course (T x 1) of the
%   mean angle between the traveling-wave velocity vectors and the +Y axis,
%   averaged across surface nodes:
%
%       theta(t) = < acos( v_y / |v| ) >_nodes
%
%   v is the struct returned by compute_velocity_field / phaseflow_cnem, with
%   fields vyp (y-component) and vnormp (speed), each T x N. The angle is in
%   radians, range [0, pi].
%
%   theta = COMPUTE_VELOCITY_ANGLE(v, smoothWin) gaussian-smooths the time
%   course with the given window (default 30).
%
%   This is the "mean angle to the A-P (Y) axis" calculation used in the
%   manuscript (extracted from meanVelocityAngle.m -> plots_meanAngleToYAxis.m).

    if nargin < 2 || isempty(smoothWin), smoothWin = 30; end
    theta = smoothdata(mean(acos(v.vyp ./ v.vnormp), 2), "gaussian", smoothWin);
end

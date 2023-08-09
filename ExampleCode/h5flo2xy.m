function [xyFlow] = h5flo2xy(h5flow)

%% usage:
% try h5flow = h5read(retFlwoPath,['/' num2str(frame)])
%     xyFlow = h5flow2xy(h5flow);
% catch
%     xyFlow = nan(1001,1001,2);
% end



%% description
% format of h5flow is channel 1: dTheta (polar angle) in deg/secc
% channel2: dRho (great circle distance from pole, with the sphere being
% the eyeball) in deg/sec

%% to convert, get a grid of starting rho and theta

% meshgrid here is -1:1,-1:1, with the sphere center being 0,0 (viewing
% from behind).
[xx,yy] = meshgrid(linspace(-1,1,1001),linspace(-1,1,1001));
% base theta is just the polar angle of a particular (x,y).
[baseTheta, baseRho] = cart2pol(xx,yy);
% base rho is the great circle distance, first cutoff below 1 (out of
% sphere)
baseRho(baseRho>1) = nan;
% arcsin gets the gc angle, since the inputted "baseRho" here is the
% back projected component as you tilt away from the fovea (sin of the
% angle)
baseRho = baseRho*pi/4;


%% permute h5 flow so that it is [row,col,channel]
flow = h5flow;
% h5 gets saved like this for some reason
%flow = permute(flow,[3 2 1]);

%% get dTheta and dRho in units of rad/frame

p_flow = opticalFlow(flow(:,:,1),flow(:,:,2));
% convert to rad/frame from deg/s
newTheta = baseTheta + deg2rad(p_flow.Vx/30);
newRho = baseRho + deg2rad(p_flow.Vy/30);

%% compute new XY positions of shifted points
[newX,newY] = pol2cart(newTheta,newRho/(pi/4));

%% change in XY is the xyFlow
xyFlow = opticalFlow(newX - xx,newY - yy);

end

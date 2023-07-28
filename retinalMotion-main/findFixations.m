function [fixation_frames] = findFixations(porX,porY,width,height,px2mmScale,calibDist,velThresh,accThresh,rEye,lEye)

framerate=120;

debug_plot = false;

%% determine fixation frames using velocity threshold


gazeVec = [rEye.circle_3d_normal_x rEye.circle_3d_normal_y rEye.circle_3d_normal_z];

gazeVec_prev = [gazeVec(1,:);gazeVec(1:end-1,:)];

gazeVec = normr(gazeVec);
gazeVec_prev = normr(gazeVec_prev);


% angle between sequential gaze directions

angVel = 2*atan2((vecnorm((gazeVec_prev - gazeVec)')'),...
    (vecnorm((gazeVec_prev + gazeVec)'))');

angVel = rad2deg(angVel)*30;

angAcc = [0 ;diff(angVel)];





fixation_frames = abs(movmean(angAcc,3))< accThresh;

fixation_frames = and(fixation_frames,movmean(angVel,3)<velThresh);


lone_dexer = find(fixation_frames);

for ii = 2:length(lone_dexer)-1
    
    if lone_dexer(ii-1)~=lone_dexer(ii)-1&&lone_dexer(ii+1)~=lone_dexer(ii)+1
        fixation_frames(lone_dexer(ii))=0;
    end
    
end


saccade_frames = ~fixation_frames;






if debug_plot
    %%
    porX_ff_plot = rEye.norm_pos_x;
    porY_ff_plot = rEye.norm_pos_y;
    
    porX_ff_plot(saccade_frames) = nan;
    porY_ff_plot(saccade_frames) = nan;
    
    angVel_ff_plot = angVel;
    angVel_ff_plot(saccade_frames) = nan;
    
    figure(3)
    clf
    subplot(3,1,1)
    plot(rEye.norm_pos_y);
    hold on
    
    plot(rEye.norm_pos_x,'color','g');
    plot(porX_ff_plot,'r.-','linewidth',2);
    plot(porY_ff_plot,'r.-','linewidth',2);
    xlim([13382 13722]);
    subplot(3,1,2);
    plot(angVel);
    hold on
    plot(angVel_ff_plot,'r.-','linewidth',2);
    xlim([13382 13722]);
    
    
    
end

end

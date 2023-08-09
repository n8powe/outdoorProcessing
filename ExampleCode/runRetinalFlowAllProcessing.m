function runRetinalFlowAllProcessing()

%% Set this to false if you have already undistorted the data and averaged the gaze data. 
preprocessVideoAndGaze = true;
chopVideo = true;
runOF = true;
runMain = true;

%% Be sure all of your paths are set correctly below here. 

if preprocessVideoAndGaze
%% Run undistortVideo.py with correct paths to video and gaze data
    pythonCall = "python";
    pythonScriptName = "undistortVideo.py";
    videoFilePath = "FullDataSet/raw-data-export/2023-07-25/rawVid.mp4";
    gazeFilePath = "FullDataSet/raw-data-export/2023-07-25/gaze.csv";
    sceneCameraJsonPath = "FullDataSet/raw-data-export/2023-07-25/scene_camera.json";
    arguments = [pythonCall, pythonScriptName, videoFilePath, gazeFilePath, sceneCameraJsonPath];
    system(join(arguments));

%% Set paths to video, gaze, info.json, and world timestamps in .config files using undistorted video and gaze data from previous step. 
%% I don't think I am going to use the config file. 

%% Run getAverageGazePositionWithinFrame.py using undistorted data. 
    pythonScriptName = "averageGazePositionPerFrame.py";
    gazeFilePath = "gaze_undist.csv";
    timePath = "FullDataSet/raw-data-export/2023-07-25/world_timestamps.csv";
    arguments_step2 = [pythonCall, pythonScriptName, gazeFilePath, timePath];
    system(join(arguments_step2));

end


%% Main Analyses below here.


%% Run createVideoSection.m using start and end frames you wish to parse. 
if chopVideo
    fullVideoPath = 'rawVid_undistorted.mp4';
    gazeDataPath = 'gaze_undist.csv';
    startFrame = 10100;
    endFrame = 10300;
    createVideoSection(fullVideoPath, gazeDataPath, startFrame, endFrame)
end

%% Run openCVOF2.py
if runOF
    pythonScriptName = "openCVOF2.py";
    outputPath = "OpticFlowFiles";
    videoPath = "choppedData/choppedVideo.mp4";
    maxFrames = 9999;
    resolution = strcat(num2str(1200),'x',num2str(1600)); % Make sure the resolution is set to what the video was recorded at. 
    arguments = [pythonCall, pythonScriptName, outputPath, videoPath, maxFrames, resolution];
    system(join(arguments));
end

%% Run main.m

if runMain
    video_path = ['choppedData/choppedVideo.mp4']; % path directly to video (undistorted)
    
    % pull from Rocks.mat / Woodchips.mat
    %data_path = ['../' sub_str  '.csv'];
    subj = 1;
    data_path = ['choppedData/choppedGazeData.csv'];
    flow_path = ['h5file.h5']; % path to .flo files
    ret_flow_path = ['retflow_flo/']; %path to output matlab flow files (retinal ref)
    sub_str = ['S' pad(num2str(subj),2,'left','0')];
    img_out_path = [sub_str '/ret_img/']; %path to output retinal images (good for visualizations)
    
    main(subj, video_path, data_path, flow_path, ret_flow_path, img_out_path);
end

%% add a visualization function down here that also saves the video. 

%flowFromVideo();

end
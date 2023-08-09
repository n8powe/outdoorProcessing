function createVideoSection(fullVideoPath, gazeDataPath, startFrame, endFrame)
    %% Run this on undistorted Data

    mkdir 'choppedData'

    v = VideoReader(fullVideoPath);

    v2 = VideoWriter('choppedData/choppedVideo.mp4', 'MPEG-4');
    v2.Quality = 100;

    open(v2)

    for f=startFrame:endFrame
        vidFrame = read(v,f);


        writeVideo(v2, vidFrame);





    end


    close(v2);
    gazeData = readtable(gazeDataPath);

    choppedGazeData = gazeData(startFrame:endFrame, :);
    writetable(choppedGazeData, 'choppedData/choppedGazeData.csv');

end
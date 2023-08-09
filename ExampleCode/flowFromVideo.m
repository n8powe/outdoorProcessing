%flow processing for vrdata
%system('ffmpeg -y -i NP_ret.mp4 -filter:v fps=30 -vf scale=640:480 thisVid.mp4');

vidReader = VideoReader("NP_ret.mp4",'CurrentTime',2);
lastFrame = [];
frameDistance = 6;
opticFlow = opticalFlowFarneback
%opticFlow.MaxIteration=10

count = 0;
while hasFrame(vidReader)

    frameRGB = readFrame(vidReader);
    frameGray = im2gray(frameRGB);  
    flow = estimateFlow(opticFlow,frameGray);

    imshow(frameRGB)

    hold on;
    plot(size(frameRGB,1)/2, size(frameRGB,2)/2, 'ro', 'MarkerSize',20)
    hold off;

    hold on
    plot(flow,'DecimationFactor',[15 15],'ScaleFactor',2);
    hold off

%     thisFrame = im2double(frameGray);
%     opticalFlow(thisFrame,lastFrame);
%     lastFrame = thisFrame;

%     imshow(frameRGB)
%     hold on
%     plot(flow,'DecimationFactor',[25 25],'ScaleFactor',500);
%     hold off

%     thisFrame = im2double(frameGray);
%     if(isempty(lastFrame)) 
%         for i = 1:frameDistance+1
%             lastFrame{i} = thisFrame;
%         end 
%     end
% 
%     opticalFlow(thisFrame,lastFrame{mod(count-frameDistance,frameDistance+1)+1});
%     mod(count-frameDistance,frameDistance+1)+1
% 
%     lastFrame{mod(count,frameDistance+1)+1} = thisFrame;

%     hsv(:,:,1) = (flow.Orientation + pi) / (2*pi);
%     hsv(:,:,2) = ones(1080,1920);
%     hsv(:,:,3) = flow.Magnitude;
%     rgb = hsv2rgb(hsv);
% 
%     imagesc(rgb);
%     drawnow;
    
    count = count+1;

    pause(10^-3)
end
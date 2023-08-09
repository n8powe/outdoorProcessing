# Run order

1) Run undistortVideo.py with correct paths to video and gaze data

2) Set paths to video, gaze data, info.json, and world_timestamps in the .config file based on **undistorted** video and gaze data.

3) Run getAverageGazePositionWithinFrame.py using the undistorded data.

4) Run createVideoSection.m -- add in the frames Start and End that you want to chop (skip this step if you are running the entire video)

5) Run openCVOF2.py using the command [python openCVOF2.py "opticFlowFiles" "choppedData/choppedVideo.mp4" 9999 1200x1600]
    Arg[1] = output folder path
    Arg[2] = video path
    Arg[3] = maxNumFrames
    Arg[4] = Optic Flow resolution to record at (I think full resolution is necessary for retinal flow? -- NP check this.)

    This will write the optic flow to .flo files.

6) Next run main.m in the retinalMotion-main folder with the path variables correctly changed.
    NOTE: Make sure the optic flow input folder points to the .flo files made in the previous step.
    NOATE: Some functions require additional modules to be downloaded from matlab add-ons. 







NOTE: The output from pupil neon doesn't save the eye states right now (august 2023). They said on their discord server that they are expecting that by Q3 2023, so hopefully soon. That seems to be necessary for fixation detection in main.m in step (6).

NOTE: **MAKE THESE CHANGES TO UNDIST GAZE** To do this, you need to estimate the focal length of the world facing camera, and then you can figure out camera relative 3d gaze vectors (the 2d eye positions can then be thought of as being on the image plane of the camera, at a focal length's distance away from the camera center.

You can use this matlab script to estimate camera params from a set of images of a checkerboard (I think there should be a giant one laying around somewhere in the lab).

You then take gazeX, gazeY (pixel coordinates minus camera center so resX/2, resY/2), add the focal length in pixels as the Z coordinate, and then normalize that vector

**ANOTHER NOTE**: I think I got the above working. But I need to rework the code that writes the flow (.flo) files so that they are written to an h5 file format. **I think I got this finished too**

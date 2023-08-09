import json
import numpy as np
#with open("scene_camera.json", "r") as f:
#  data = json.load(f)

#camera_matrix = np.array(data["camera_matrix"])
#dist_coeffs = np.array(data["distortion_coefficients"])

#print("Camera Matrix:")
#print(camera_matrix)
#print("Distortion Coefficients:")
#print(dist_coeffs)



import pathlib

import av
from tqdm import tqdm
import pandas as pd
import cv2
import os
import sys

def undistort_video(
    original_video_path, undistorted_video_path, scene_camera_path
):
    timestamps_path = pathlib.Path(original_video_path).with_name(
        "world_timestamps.csv"
    )
    num_frames = pd.read_csv(timestamps_path).shape[0]
    original_container = av.open(str(original_video_path))
    original_video_stream = original_container.streams.video[0]

    undistorted_container = av.open(str(undistorted_video_path), "w")

    with open(scene_camera_path, "r") as f:
      data = json.load(f)

    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["distortion_coefficients"])

    try:
        undistorted_video = undistorted_container.add_stream("h264_nvenc")
    except Exception as e:
        print("nvenc not available", e)
        undistorted_video = undistorted_container.add_stream("h264")

    undistorted_video.options["bf"] = "0"
    undistorted_video.options["movflags"] = "faststart"
    undistorted_video.gop_size = original_video_stream.gop_size
    undistorted_video.codec_context.height = original_video_stream.height
    undistorted_video.codec_context.width = original_video_stream.width
    undistorted_video.codec_context.time_base = original_video_stream.time_base
    undistorted_video.codec_context.bit_rate = original_video_stream.bit_rate

    if original_container.streams.audio:
        audio_stream = original_container.streams.audio[0]
        output_audio_stream = undistorted_container.add_stream("aac")
        output_audio_stream.codec_context.layout = audio_stream.layout.name
        output_audio_stream.codec_context.time_base = audio_stream.time_base
        output_audio_stream.codec_context.bit_rate = audio_stream.bit_rate
        output_audio_stream.codec_context.sample_rate = audio_stream.sample_rate

    progress = tqdm(unit=" frames", total=num_frames)
    with undistorted_container:
        for packet in original_container.demux():
            frames = packet.decode()

            if packet.stream.type == "audio":
                for frame in frames:
                    packets = output_audio_stream.encode(frame)
                    undistorted_container.mux(packets)
            elif packet.stream.type == "video":
                for frame in frames:
                    #print ("Frame---", frame)
                    img = frame.to_ndarray(format="bgr24")
                    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
                    new_frame = frame.from_ndarray(undistorted_img, format="bgr24")
                    new_frame.pts = frame.pts
                    new_frame.time_base = original_video_stream.time_base
                    packets = undistorted_video.encode(new_frame)
                    progress.update()
                    undistorted_container.mux(packets)
        # encode and mux frames that have been queued internally by the encoders
        undistorted_container.mux(output_audio_stream.encode())
        undistorted_container.mux(undistorted_video.encode())


def undistort_gaze(original_gaze_path, unditorted_gaze_path, scene_camera_path):

    with open(scene_camera_path, "r") as f:
      data = json.load(f)

    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["distortion_coefficients"])

    original_gaze_df = pd.read_csv(original_gaze_path)
    original_gaze = original_gaze_df[["gaze_x_px", "gaze_y_px"]].values
    undistorted_gaze = cv2.undistortPoints(
        original_gaze.reshape(-1, 2), camera_matrix, dist_coeffs, P=camera_matrix
    )

    undistorted_gaze_df = original_gaze_df.copy()
    undistorted_gaze_df[["gaze_x_px", "gaze_y_px"]] = undistorted_gaze.reshape(-1, 2)
    undistorted_gaze_df[["gaze_z_px"]] = np.ones([len(undistorted_gaze_df[["gaze_x_px"]]),1])
    undistorted_gaze_df.to_csv(unditorted_gaze_path, index=False)



# Where we get the video(s)



recording_folder = ""
scene_camera_path = sys.argv[3]
original_video_path = sys.argv[1]
undistorted_video_path = recording_folder + "rawVid_undistorted.mp4"
undistort_video(original_video_path, undistorted_video_path, scene_camera_path)

original_gaze_path = sys.argv[2]
undistorted_gaze_path = recording_folder + "gaze_undist.csv"
undistort_gaze(original_gaze_path, undistorted_gaze_path, scene_camera_path)

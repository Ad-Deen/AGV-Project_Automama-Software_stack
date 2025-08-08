#!/usr/bin/env python3

from jetson_inference import segNet
from jetson_utils import videoSource, videoOutput, cudaDeviceSynchronize

# Load the segmentation network
net = segNet("fcn-resnet18-voc")
net.SetOverlayAlpha(150.0)

# Correct way to load a video file
video_path = "file:///home/deen/ros2_ws/src/automama/automama/perception/test_vids/video8.mp4"
input_video = videoSource(video_path)
output = videoOutput("display://0")  # Render to screen

# Run segmentation on the video
while input_video.IsStreaming() and output.IsStreaming():
    frame = input_video.Capture()

    if frame is None:
        continue

    net.Process(frame)
    net.Overlay(frame)

    output.Render(frame)
    output.SetStatus("FPS: {:.2f}".format(net.GetNetworkFPS()))

    cudaDeviceSynchronize()

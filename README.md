# Gesture Recognition

Uses MediaPipe hands to track the following gestures:

- "Forwards": palm is open and moving towards the camera
- Others to be decided

## `theta_streamer`

Getting GStreamer to work with OpenCV in Python is a pain.

This helper program is receives frames from a GStreamer Pipeline in C (which is easier to get up and running, hilariously enough), then writes those frames to a shared memory buffer that may be read by the Python application.

This keeps latency low, as the process running `theta_streamer` will focus purely on receiving frames and writing them to the buffer, and the python process can focus purely on analyzing the most recently read frame.

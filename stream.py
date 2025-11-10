import cv2
import mediapipe as mp
import mmap
import numpy as np
import os
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

base_options = mp_tasks.BaseOptions(
    model_asset_path="./pose_landmarker_lite.task",
    delegate=mp_tasks.BaseOptions.Delegate.GPU
)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

SHBUFFER_SIZE = 1920 * 960 * 3

mp_drawing = mp.solutions.drawing_utils

def main():
    fd = os.open("/dev/shm/theta_stream", os.O_RDWR);
    shbuffer = mmap.mmap(fd, SHBUFFER_SIZE, access=mmap.ACCESS_READ);
    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
        )
    )

    while True:
        buffer = np.frombuffer(shbuffer, dtype=np.uint8)
        frame = buffer.reshape((960, 1920, 3)).copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        pose_results = detector.detect(mp_image)

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"Press 'q' to exit", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

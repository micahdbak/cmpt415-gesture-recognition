import cv2
import math
import mediapipe as mp
import numpy as np
from enum import Enum

class GestureDirection(Enum):
    LEFT = 1
    FORWARD = 2
    RIGHT = 3
    UNKNOWN = 4

# gets capture devices and shows resolution
# (360 cam has resolution of 1920 x 960)
def prepare_capture():
    devices = []

    # find all video capture indices 0..10
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)  # macOS backend
        if cap.isOpened():
            devices.append({
                "index": i,
                "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            })
            cap.release()
        else:
            break

    if not devices:
        print("No video devices connected!")
        return

    print("----\n(Ignore previous output.)\nDevices:")
    for device in devices:
        print(f"\t{device['index']}:\t{int(device['width'])} x {int(device['height'])}")

    device = int(input(f"Choose device: ").strip())

    cap = cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Failed to open device {device}.")
        return
    
    return cap

# returns two angles: [theta0, theta1]
# theta0: rotation of hand relative to camera
def get_palm_features(frame, hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    pts = []
    tri = []

    # use wrist as reference (origin) point
    origin = [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z]

    # 0: wrist, 5: palm x index, 17: palm x pinky
    for idx in [0, 5, 17]:
        lm = hand_landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)

        pts.append([x, y])
        tri.append([lm.x - origin[0], lm.y - origin[1], lm.z - origin[2]])

    # display points for reference
    cv2.polylines(frame, [np.array(pts, np.int32)], isClosed=True, color=(0, 0, 0), thickness=5)

    # v1: middle point between index and pinky points
    v1 = np.array([(tri[1][0] + tri[2][0])/2,
                   (tri[1][1] + tri[2][1])/2,
                   (tri[1][2] + tri[2][2])/2])
    # v2: reference vector, which points directly upwards
    v2 = np.array([0.0, 1.0, 0.0])

    # no hand information
    if v1[0] == v1[1] == v1[2] == 0.0:
        return [GestureDirection.UNKNOWN, 0.0]

    # theta0 represents the rotation of the hand *around* v2
    theta0 = np.arctan2(v1[2], v1[0])
    # theta1 represents the angle between the hand and v2
    theta1 = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))

    direction = GestureDirection.UNKNOWN

    if theta0 > 0.0:
        direction = GestureDirection.UNKNOWN
    elif theta0 > -(math.pi / 4.0):
        direction = GestureDirection.LEFT
    elif theta0 > -((3.0 * math.pi) / 4.0):
        direction = GestureDirection.FORWARD
    elif theta0 > -math.pi:
        direction = GestureDirection.RIGHT

    if direction == GestureDirection.UNKNOWN:
        direction = get_palm_features.last_direction
    else:
        get_palm_features.last_direction = direction

    return [direction, theta1]
get_palm_features.last_direction = GestureDirection.FORWARD

def render_theta(frame, theta, cx, cy):
    L = 32 # line length in pixels

    x1 = cx
    y1 = cy - L
    x2 = int(cx + L * math.sin(theta))
    y2 = int(cy - L * math.cos(theta))

    cv2.line(frame, (cx, cy), (x1, y1), (0, 0, 0), 2)
    cv2.line(frame, (cx, cy), (x2, y2), (255, 0, 0), 2)

def main():
    cap = prepare_capture()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Instantiate the hands module
    hands = mp_hands.Hands(
        static_image_mode=False,       # For video feed; True for single images
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        features = [GestureDirection.UNKNOWN, 0.0]

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # features = get_palm_features(frame, hand_landmarks, frame.shape)

                # z_values = [lm.z for lm in hand_landmarks.landmark]
                # avg_z = sum(z_values) / len(z_values)
                # print(f"Relative depth: {avg_z:.3f}")

        # render_theta(frame, features[0], 64, 80)
        # render_theta(frame, features[1], 64, 80)
        
        if features[1] > math.pi / 2.0 and features[1] != GestureDirection.UNKNOWN:
            print(f"Gesturing towards {features[0]}")

        cv2.imshow(f"Press 'q' to exit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

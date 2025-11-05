# hand_detection.py
import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(0)  # change 0 -> 1 if your webcam is on another index
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        prev_time = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Couldn't read from camera. Check camera index or permissions.")
                break

            # Mirror image so it feels like a webcam selfie
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                # results.multi_hand_landmarks and results.multi_handedness align by index
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Draw landmarks and connections
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get left/right label and score
                    label = handedness.classification[0].label
                    score = handedness.classification[0].score

                    # Build bounding box from normalized landmark coords
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    x_min = int(min(x_coords) * w) - 20
                    x_max = int(max(x_coords) * w) + 20
                    y_min = int(min(y_coords) * h) - 20
                    y_max = int(max(y_coords) * h) + 20
                    x_min, y_min = max(x_min, 0), max(y_min, 0)
                    x_max, y_max = min(x_max, w), min(y_max, h)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {score:.2f}', (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Example: mark index fingertip (landmark 8)
                    ix = int(hand_landmarks.landmark[8].x * w)
                    iy = int(hand_landmarks.landmark[8].y * h)
                    cv2.circle(frame, (ix, iy), 6, (255, 0, 0), cv2.FILLED)
                    cv2.putText(frame, f'({ix},{iy})', (ix + 8, iy + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # FPS counter
            cur_time = time.time()
            fps = 1.0 / (cur_time - prev_time) if prev_time else 0.0
            prev_time = cur_time
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Hand Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                break
            elif key == ord('s'):
                cv2.imwrite('hand_capture.png', frame)
                print("Saved hand_capture.png")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

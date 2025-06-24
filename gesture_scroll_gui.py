import cv2
import mediapipe as mp
import pyautogui
import time

last_scroll_time = 0
scroll_cooldown = 0.5  

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

last_avg_y = None

SCROLL_THRESHOLD = 0.015 
LAST_SCROLL_TIME = time.time()
SCROLL_COOLDOWN_SECONDS = 0.05 
def get_hand_gesture(hand_landmarks_list):
    
    global last_avg_y, LAST_SCROLL_TIME

    
    index_finger_tip_y = hand_landmarks_list.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_finger_tip_y = hand_landmarks_list.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    index_finger_pip_y = hand_landmarks_list.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_finger_pip_y = hand_landmarks_list.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

    index_extended = index_finger_tip_y < index_finger_pip_y
    middle_extended = middle_finger_tip_y < middle_finger_pip_y

    if index_extended and middle_extended:
        avg_y = (index_finger_tip_y + middle_finger_tip_y) / 2

        if last_avg_y is not None:
            delta_y = last_avg_y - avg_y

            current_time = time.time()
            if current_time - LAST_SCROLL_TIME > SCROLL_COOLDOWN_SECONDS:
                if delta_y > SCROLL_THRESHOLD:
                    pyautogui.scroll(30) # Scroll up
                    LAST_SCROLL_TIME = current_time
                    return 'Two-finger Scroll Up'
                elif delta_y < -SCROLL_THRESHOLD:
                    pyautogui.scroll(-30) # Scroll down
                    LAST_SCROLL_TIME = current_time
                    return 'Two-finger Scroll Down'
                else:
                    return 'Two-finger Hold'
            else:
                return 'Two-finger Hold (Cooldown)' 
        
        last_avg_y = avg_y
        return 'Two-finger Hold' 
    else:
        last_avg_y = None
        return 'Fingers Not Extended'

def main():
    """
    Main function to run the hand gesture detection and scrolling application.
    """
    # webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_hands.Hands(
        model_complexity=1, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

        print("Webcam started. Move your hand into view.")
        print("Extend your index and middle fingers to scroll.")
        print("Press 'q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image.flags.writeable = False

            results = hands.process(image)

            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            current_gesture_status = "No Hand Detected"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    current_gesture_status = get_hand_gesture(hand_landmarks)
                    break 

            cv2.putText(image, f"Gesture: {current_gesture_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Hand Gesture Scroll Control', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # Release 
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    main()

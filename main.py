import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ─────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────
SMOOTHING          = 0.12
SCROLL_SPEED       = 30
CLICK_COOLDOWN     = 0.4
FRAME_REDUCTION    = 70
SCROLL_THRESHOLD   = 0.03
TAP_THRESHOLD      = 0.08   # how bent index must be to count as a tap
TAP_WINDOW         = 0.6    # seconds within which 2 taps = double tap click
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.7,
)

screen_w, screen_h = pyautogui.size()

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def is_finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def is_thumb_up(lm):
    thumb_up    = lm[4].y < lm[3].y - 0.04
    index_down  = not is_finger_up(lm, 8,  6)
    middle_down = not is_finger_up(lm, 12, 10)
    ring_down   = not is_finger_up(lm, 16, 14)
    pinky_down  = not is_finger_up(lm, 20, 18)
    return thumb_up and index_down and middle_down and ring_down and pinky_down

def is_index_bent(lm):
    """Index finger is bent when tip is BELOW the pip joint."""
    return lm[8].y > lm[6].y + TAP_THRESHOLD

def two_fingers_up(lm):
    return (
        is_finger_up(lm, 8,  6)  and
        is_finger_up(lm, 12, 10) and
        not is_finger_up(lm, 16, 14) and
        not is_finger_up(lm, 20, 18)
    )

def detect_gesture(lm):
    index_up  = is_finger_up(lm, 8,  6)
    middle_up = is_finger_up(lm, 12, 10)
    ring_up   = is_finger_up(lm, 16, 14)
    pinky_up  = is_finger_up(lm, 20, 18)

    if is_thumb_up(lm):
        return 'right_click'

    if index_up and not middle_up and not ring_up and not pinky_up:
        return 'move'

    if two_fingers_up(lm):
        return 'scroll_mode'

    return 'none'

def draw_overlay(frame, gesture, scroll_dir, tap_count, fps):
    h, w = frame.shape[:2]

    display = gesture
    color   = (100, 100, 100)

    if gesture == 'move':
        if tap_count == 1:
            display = 'move (1 tap...)'
            color   = (100, 200, 255)
        else:
            color   = (50, 200, 50)
    elif gesture == 'click':
        display = 'LEFT CLICK!'
        color   = (50, 50, 255)
    elif gesture == 'right_click':
        color   = (200, 50, 200)
    elif gesture == 'scroll_mode':
        if scroll_dir == 'up':
            display = 'scroll UP'
            color   = (50, 200, 200)
        elif scroll_dir == 'down':
            display = 'scroll DOWN'
            color   = (200, 200, 50)
        else:
            display = 'scroll (move hand)'
            color   = (150, 150, 150)

    label = f"  {display.upper()}  "
    cv2.rectangle(frame, (10, 10), (340, 50), (0, 0, 0), -1)
    cv2.putText(frame, label, (14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    instructions = [
        "Q                   -> Quit",
        "2 fingers + move dn -> Scroll down",
        "2 fingers + move up -> Scroll up",
        "Thumb only up       -> Right click",
        "Index double tap    -> Left click",
        "Index only up       -> Move cursor",
    ]
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, h - 20 - i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_x, prev_y   = 0, 0
    last_click_time   = 0
    prev_time         = time.time()
    prev_index_y      = None
    scroll_dir        = None

    # Double tap tracking
    index_was_bent    = False   # was index bent in previous frame?
    tap_count         = 0       # how many taps detected
    first_tap_time    = 0       # time of first tap

    print("Hand Mouse Control running!")
    print("Show your hand to the camera.")
    print("Press Q to quit.\n")
    print("LEFT CLICK  = raise index finger then quickly bend & straighten it TWICE")
    print("RIGHT CLICK = thumb only up\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture    = 'none'
        scroll_dir = None
        now        = time.time()

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                frame, hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            lm      = hand_lm.landmark
            gesture = detect_gesture(lm)

            # Cursor position
            raw_x = np.interp(lm[8].x, [FRAME_REDUCTION/w, 1 - FRAME_REDUCTION/w], [0, screen_w])
            raw_y = np.interp(lm[8].y, [FRAME_REDUCTION/h, 1 - FRAME_REDUCTION/h], [0, screen_h])
            smooth_x = prev_x + (raw_x - prev_x) * SMOOTHING
            smooth_y = prev_y + (raw_y - prev_y) * SMOOTHING
            prev_x, prev_y = smooth_x, smooth_y

            # ── Double tap detection (only in move mode) ──────────
            if gesture == 'move':
                index_bent_now = is_index_bent(lm)

                # Detect a bend → straighten transition = one tap
                if index_was_bent and not index_bent_now:
                    tap_count += 1
                    if tap_count == 1:
                        first_tap_time = now

                index_was_bent = index_bent_now

                # Reset if tap window expired
                if tap_count > 0 and (now - first_tap_time) > TAP_WINDOW:
                    tap_count = 0

                # Two taps within window = left click
                if tap_count >= 2:
                    if now - last_click_time > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_click_time = now
                        gesture = 'click'
                    tap_count = 0

                pyautogui.moveTo(smooth_x, smooth_y)

            else:
                # Reset tap state when not in move mode
                index_was_bent = False
                tap_count      = 0

            # ── Right click ───────────────────────────────────────
            if gesture == 'right_click':
                pyautogui.moveTo(smooth_x, smooth_y)
                if now - last_click_time > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_click_time = now

            # ── Scroll mode ───────────────────────────────────────
            elif gesture == 'scroll_mode':
                curr_y = lm[8].y
                if prev_index_y is not None:
                    delta = curr_y - prev_index_y
                    if delta < -SCROLL_THRESHOLD:
                        scroll_dir = 'up'
                        pyautogui.scroll(SCROLL_SPEED)
                    elif delta > SCROLL_THRESHOLD:
                        scroll_dir = 'down'
                        pyautogui.scroll(-SCROLL_SPEED)
                prev_index_y = curr_y

            else:
                prev_index_y = None

        else:
            index_was_bent = False
            tap_count      = 0
            prev_index_y   = None

        curr_time = time.time()
        fps       = 1 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time

        draw_overlay(frame, gesture, scroll_dir, tap_count, fps)

        cv2.imshow("Hand Mouse Control  |  Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Bye!")


if __name__ == "__main__":
    main()
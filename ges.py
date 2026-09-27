import cv2
import mediapipe as mp
import pyautogui
import math
import time
import ctypes
from ctypes import wintypes
from datetime import datetime


# =========================================================
# SETTINGS
# =========================================================

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


# =========================================================
# CURSOR SETTINGS
# =========================================================

CURSOR_SENSITIVITY = 3.2
CURSOR_SMOOTHING = 0.38
CURSOR_DEAD_ZONE = 0.0010
CURSOR_MAX_STEP = 55


# =========================================================
# PINKY CLICK MODE
# =========================================================

PINKY_ON_FRAMES = 5
PINKY_OFF_FRAMES = 7


# =========================================================
# TAP / CLICK SETTINGS
# =========================================================

TAP_DOWN = 0.012
TAP_UP = 0.008
TAP_MAX_TIME = 0.45

DOUBLE_TAP_TIME = 0.38
CLICK_COOLDOWN = 0.18

CLICK_NOISE = 0.002


# =========================================================
# SCROLL SETTINGS
# =========================================================

THUMB_STABLE_FRAMES = 5
SCROLL_AMOUNT = 1
THUMB_DIRECTION_THRESHOLD = 0.16
SCROLL_COOLDOWN = 0.55


# =========================================================
# EAR VOLUME
# =========================================================

EAR_DISTANCE = 0.075
EAR_TAP_COOLDOWN = 0.50
EAR_ACTION_FRAMES = 3


# =========================================================
# BLINK
# =========================================================

BLINK_THRESHOLD = 0.20
BLINK_MIN_TIME = 0.05
BLINK_MAX_TIME = 0.60
DOUBLE_BLINK_TIME = 0.80
SCREENSHOT_COOLDOWN = 1.20


# =========================================================
# WINDOWS SCROLL
# =========================================================

user32 = ctypes.windll.user32

WM_MOUSEWHEEL = 0x020A
WHEEL_DELTA = 120


# =========================================================
# PYAUTOGUI SAFETY
# =========================================================

pyautogui.PAUSE = 0.02
pyautogui.FAILSAFE = False


# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.60,
    min_hand_presence_confidence=0.60,
    min_tracking_confidence=0.60
)


FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.50,
    min_face_presence_confidence=0.50,
    min_tracking_confidence=0.50,
    output_face_blendshapes=True
)


# =========================================================
# STATE
# =========================================================

cursor_x = None
cursor_y = None

last_hand_x = None
last_hand_y = None


# =========================================================
# PINKY STATE
# =========================================================

pinky_on_counter = 0
pinky_off_counter = 0

click_mode = False


# =========================================================
# TAP STATE
# =========================================================

tap_state = "READY"

tap_baseline_y = None
tap_down_y = None
tap_down_time = 0.0

waiting_for_second_tap = False
first_tap_time = 0.0

last_click_time = 0.0


# =========================================================
# SCROLL STATE
# =========================================================

thumb_scroll_counter = 0
thumb_scroll_active = False
thumb_scroll_direction = None

last_scroll_time = 0.0


# =========================================================
# EAR STATE
# =========================================================

left_ear_counter = 0
right_ear_counter = 0

last_ear_action_time = 0.0


# =========================================================
# BLINK STATE
# =========================================================

eyes_closed = False
blink_start_time = 0.0

last_blink_time = 0.0
blink_count = 0

last_screenshot_time = 0.0

timestamp_ms = 0


# =========================================================
# MATH
# =========================================================

def distance(a, b):

    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def angle(a, b, c):

    ab = (
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    )

    cb = (
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    )

    dot = (
        ab[0] * cb[0] +
        ab[1] * cb[1] +
        ab[2] * cb[2]
    )

    mag1 = math.sqrt(
        ab[0] ** 2 +
        ab[1] ** 2 +
        ab[2] ** 2
    )

    mag2 = math.sqrt(
        cb[0] ** 2 +
        cb[1] ** 2 +
        cb[2] ** 2
    )

    if mag1 == 0 or mag2 == 0:
        return 0

    value = dot / (mag1 * mag2)
    value = max(-1.0, min(1.0, value))

    return math.degrees(
        math.acos(value)
    )


def hand_size(hand):

    return distance(
        hand[0],
        hand[9]
    )


# =========================================================
# FINGER DETECTION
# =========================================================

def finger_folded(hand, tip, pip):

    wrist = hand[0]

    tip_distance = distance(
        hand[tip],
        wrist
    )

    pip_distance = distance(
        hand[pip],
        wrist
    )

    return tip_distance < pip_distance * 1.30


def finger_up(hand, tip, pip):

    wrist = hand[0]

    tip_distance = distance(
        hand[tip],
        wrist
    )

    pip_distance = distance(
        hand[pip],
        wrist
    )

    return tip_distance > pip_distance * 1.10


# =========================================================
# PINKY DETECTION
# =========================================================

def is_pinky_up(hand):

    if not finger_up(
        hand,
        20,
        18
    ):
        return False

    a1 = angle(
        hand[17],
        hand[18],
        hand[19]
    )

    a2 = angle(
        hand[18],
        hand[19],
        hand[20]
    )

    return (
        a1 > 120 and
        a2 > 120
    )


# =========================================================
# CLICK MODE
# =========================================================

def update_click_mode(hand):

    global pinky_on_counter
    global pinky_off_counter
    global click_mode

    if is_pinky_up(hand):

        pinky_on_counter += 1
        pinky_off_counter = 0

        if pinky_on_counter >= PINKY_ON_FRAMES:

            if not click_mode:

                reset_tap()
                reset_click_sequence()

            click_mode = True

    else:

        pinky_off_counter += 1
        pinky_on_counter = 0

        if pinky_off_counter >= PINKY_OFF_FRAMES:

            if click_mode:

                reset_tap()
                reset_click_sequence()

            click_mode = False


# =========================================================
# SCROLL GESTURE
# =========================================================

def is_thumb_scroll_gesture(hand):

    index_folded = finger_folded(
        hand,
        8,
        6
    )

    middle_folded = finger_folded(
        hand,
        12,
        10
    )

    ring_folded = finger_folded(
        hand,
        16,
        14
    )

    pinky_folded = finger_folded(
        hand,
        20,
        18
    )

    if not (
        index_folded and
        middle_folded and
        ring_folded and
        pinky_folded
    ):

        return False

    size = hand_size(hand)

    if size <= 0:
        return False

    thumb_length = distance(
        hand[2],
        hand[4]
    )

    if thumb_length < size * 0.30:
        return False

    return True


# =========================================================
# THUMB DIRECTION
# =========================================================

def get_thumb_direction(hand):

    if not is_thumb_scroll_gesture(hand):
        return None

    size = hand_size(hand)

    if size <= 0:
        return None

    difference = (
        hand[4].y -
        hand[2].y
    )

    threshold = (
        size *
        THUMB_DIRECTION_THRESHOLD
    )

    if difference < -threshold:
        return "UP"

    if difference > threshold:
        return "DOWN"

    return None


# =========================================================
# RESET SCROLL
# =========================================================

def reset_scroll():

    global thumb_scroll_counter
    global thumb_scroll_active
    global thumb_scroll_direction
    global last_scroll_time

    thumb_scroll_counter = 0
    thumb_scroll_active = False
    thumb_scroll_direction = None
    last_scroll_time = 0.0


# =========================================================
# SEND SCROLL
# =========================================================

def send_scroll(direction):

    try:

        x, y = pyautogui.position()

        point = wintypes.POINT(
            int(x),
            int(y)
        )

        hwnd = user32.WindowFromPoint(
            point
        )

        if not hwnd:
            return False

        if direction == "UP":
            wheel = WHEEL_DELTA
        else:
            wheel = -WHEEL_DELTA

        wparam = (
            (wheel & 0xFFFF)
            << 16
        )

        lparam = (
            (int(y) & 0xFFFF)
            << 16
        ) | (
            int(x) & 0xFFFF
        )

        user32.PostMessageW(
            hwnd,
            WM_MOUSEWHEEL,
            wparam,
            lparam
        )

        return True

    except Exception:

        return False


# =========================================================
# PROCESS SCROLL
# =========================================================

def process_thumb_scroll(hand):

    global thumb_scroll_counter
    global thumb_scroll_active
    global thumb_scroll_direction
    global last_scroll_time

    gesture_active = is_thumb_scroll_gesture(
        hand
    )

    if not gesture_active:

        reset_scroll()

        return False, "NORMAL MODE"

    direction = get_thumb_direction(
        hand
    )

    if direction is None:

        thumb_scroll_counter = 0

        return True, "SCROLL READY"

    if thumb_scroll_direction is None:

        thumb_scroll_direction = direction
        thumb_scroll_counter = 1

        return True, "SCROLL READY"

    if direction != thumb_scroll_direction:

        thumb_scroll_direction = direction
        thumb_scroll_counter = 1
        thumb_scroll_active = False

        return True, "SCROLL READY"

    thumb_scroll_counter += 1

    if thumb_scroll_counter < THUMB_STABLE_FRAMES:

        reset_cursor()

        return True, "SCROLL READY"

    if thumb_scroll_active:

        reset_cursor()

        return True, "SCROLL HOLD"

    now = time.time()

    if (
        now - last_scroll_time
        <
        SCROLL_COOLDOWN
    ):

        reset_cursor()

        return True, "SCROLL WAIT"

    success = send_scroll(
        direction
    )

    if success:

        print(
            "ONE SCROLL:",
            direction
        )

    thumb_scroll_active = True
    last_scroll_time = now

    reset_cursor()

    return True, "SCROLL " + direction


# =========================================================
# CURSOR RESET
# =========================================================

def reset_cursor():

    global cursor_x
    global cursor_y
    global last_hand_x
    global last_hand_y

    cursor_x = None
    cursor_y = None

    last_hand_x = None
    last_hand_y = None


# =========================================================
# CURSOR MOVEMENT
# =========================================================

def move_cursor(hand):

    global cursor_x
    global cursor_y
    global last_hand_x
    global last_hand_y

    index = hand[8]

    x = index.x
    y = index.y

    if cursor_x is None:

        cursor_x = x
        cursor_y = y

        last_hand_x = x
        last_hand_y = y

        return

    cursor_x += (
        x - cursor_x
    ) * CURSOR_SMOOTHING

    cursor_y += (
        y - cursor_y
    ) * CURSOR_SMOOTHING

    dx = (
        cursor_x -
        last_hand_x
    )

    dy = (
        cursor_y -
        last_hand_y
    )

    last_hand_x = cursor_x
    last_hand_y = cursor_y

    if abs(dx) < CURSOR_DEAD_ZONE:
        dx = 0.0

    if abs(dy) < CURSOR_DEAD_ZONE:
        dy = 0.0

    if dx == 0.0 and dy == 0.0:
        return

    screen_w, screen_h = pyautogui.size()

    move_x = int(
        dx *
        screen_w *
        CURSOR_SENSITIVITY
    )

    move_y = int(
        dy *
        screen_h *
        CURSOR_SENSITIVITY
    )

    move_x = max(
        -CURSOR_MAX_STEP,
        min(
            CURSOR_MAX_STEP,
            move_x
        )
    )

    move_y = max(
        -CURSOR_MAX_STEP,
        min(
            CURSOR_MAX_STEP,
            move_y
        )
    )

    if move_x == 0 and move_y == 0:
        return

    current_x, current_y = (
        pyautogui.position()
    )

    new_x = max(
        2,
        min(
            screen_w - 2,
            current_x + move_x
        )
    )

    new_y = max(
        2,
        min(
            screen_h - 2,
            current_y + move_y
        )
    )

    pyautogui.moveTo(
        new_x,
        new_y,
        duration=0
    )


# =========================================================
# CLICK RESET
# =========================================================

def reset_tap():

    global tap_state
    global tap_baseline_y
    global tap_down_y
    global tap_down_time

    tap_state = "READY"

    tap_baseline_y = None
    tap_down_y = None
    tap_down_time = 0.0


def reset_click_sequence():

    global waiting_for_second_tap
    global first_tap_time

    waiting_for_second_tap = False
    first_tap_time = 0.0


# =========================================================
# CLICK ACTION
# =========================================================

def do_left_click():

    global last_click_time

    now = time.time()

    if now - last_click_time < CLICK_COOLDOWN:
        return

    # =====================================================
    # 1 AIR TAP = DOUBLE LEFT CLICK
    # =====================================================

    pyautogui.doubleClick(
        interval=0.08,
        button="left"
    )

    last_click_time = now

    print(
        "ONE AIR TAP - OPEN / DOUBLE CLICK"
    )


def do_right_click():

    global last_click_time

    now = time.time()

    if now - last_click_time < CLICK_COOLDOWN:
        return

    pyautogui.click(
        button="right"
    )

    last_click_time = now

    print(
        "RIGHT CLICK"
    )


# =========================================================
# REGISTER AIR TAP
# =========================================================

def register_tap():

    global waiting_for_second_tap
    global first_tap_time

    now = time.time()

    # =====================================================
    # FIRST TAP
    # =====================================================

    if not waiting_for_second_tap:

        waiting_for_second_tap = True
        first_tap_time = now

        print(
            "AIR TAP 1"
        )

        return

    # =====================================================
    # SECOND TAP = RIGHT CLICK
    # =====================================================

    elapsed = now - first_tap_time

    if elapsed <= DOUBLE_TAP_TIME:

        do_right_click()

        waiting_for_second_tap = False
        first_tap_time = 0.0

        return

    # =====================================================
    # PREVIOUS SINGLE TAP = OPEN
    # =====================================================

    do_left_click()

    waiting_for_second_tap = True
    first_tap_time = now


# =========================================================
# SINGLE TAP TIMER
# =========================================================

def check_single_click():

    global waiting_for_second_tap
    global first_tap_time

    if not waiting_for_second_tap:
        return

    now = time.time()

    elapsed = now - first_tap_time

    if elapsed >= DOUBLE_TAP_TIME:

        # =================================================
        # ONE AIR TAP = OPEN
        # =================================================

        do_left_click()

        waiting_for_second_tap = False
        first_tap_time = 0.0


# =========================================================
# NATURAL AIR TAP
# =========================================================

def process_click(hand):

    global tap_state
    global tap_baseline_y
    global tap_down_y
    global tap_down_time

    index_tip = hand[8]

    current_y = index_tip.y

    now = time.time()

    # -----------------------------------------------------
    # CHECK PENDING SINGLE TAP
    # -----------------------------------------------------

    check_single_click()

    # -----------------------------------------------------
    # HAND SIZE
    # -----------------------------------------------------

    size = hand_size(hand)

    if size <= 0:
        return

    # -----------------------------------------------------
    # INITIAL BASELINE
    # -----------------------------------------------------

    if tap_baseline_y is None:

        tap_baseline_y = current_y
        tap_state = "READY"

        return

    # =====================================================
    # READY
    # =====================================================

    if tap_state == "READY":

        movement = (
            current_y -
            tap_baseline_y
        )

        # -------------------------------------------------
        # IGNORE SMALL MOVEMENT
        # -------------------------------------------------

        if abs(movement) < CLICK_NOISE:

            tap_baseline_y = (
                tap_baseline_y * 0.96
                +
                current_y * 0.04
            )

            return

        # -------------------------------------------------
        # DOWNWARD TAP
        # -------------------------------------------------

        if movement >= TAP_DOWN:

            tap_state = "DOWN"

            tap_down_y = current_y
            tap_down_time = now

            return

        # -------------------------------------------------
        # BASELINE FOLLOW
        # -------------------------------------------------

        if abs(movement) < 0.040:

            tap_baseline_y = (
                tap_baseline_y * 0.90
                +
                current_y * 0.10
            )

            return

    # =====================================================
    # DOWN
    # =====================================================

    elif tap_state == "DOWN":

        # -------------------------------------------------
        # CANCEL LONG HOLD
        # -------------------------------------------------

        if now - tap_down_time > TAP_MAX_TIME:

            tap_state = "READY"

            tap_down_y = None
            tap_down_time = 0.0

            tap_baseline_y = current_y

            return

        # -------------------------------------------------
        # RETURN UP
        # -------------------------------------------------

        upward_movement = (
            tap_down_y -
            current_y
        )

        # -------------------------------------------------
        # TAP COMPLETE
        # -------------------------------------------------

        if upward_movement >= TAP_UP:

            print(
                "TAP DETECTED"
            )

            register_tap()

            tap_state = "READY"

            tap_down_y = None
            tap_down_time = 0.0

            tap_baseline_y = current_y

            return


# =========================================================
# FACE DISTANCE
# =========================================================

def landmark_distance(face, a, b):

    return math.sqrt(
        (face[a].x - face[b].x) ** 2 +
        (face[a].y - face[b].y) ** 2
    )


# =========================================================
# EYE ASPECT RATIO
# =========================================================

def eye_aspect_ratio(
    face,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6
):

    vertical_1 = landmark_distance(
        face,
        p2,
        p6
    )

    vertical_2 = landmark_distance(
        face,
        p3,
        p5
    )

    horizontal = landmark_distance(
        face,
        p1,
        p4
    )

    if horizontal == 0:
        return 1.0

    return (
        vertical_1 +
        vertical_2
    ) / (
        2.0 *
        horizontal
    )


# =========================================================
# BLINK SCREENSHOT
# =========================================================

def process_blink(face):

    global eyes_closed
    global blink_start_time
    global last_blink_time
    global blink_count
    global last_screenshot_time

    left_eye = eye_aspect_ratio(
        face,
        33,
        160,
        158,
        133,
        153,
        144
    )

    right_eye = eye_aspect_ratio(
        face,
        362,
        385,
        387,
        263,
        373,
        380
    )

    eye_ratio = (
        left_eye +
        right_eye
    ) / 2.0

    now = time.time()

    if eye_ratio < BLINK_THRESHOLD:

        if not eyes_closed:

            eyes_closed = True
            blink_start_time = now

    else:

        if eyes_closed:

            duration = (
                now -
                blink_start_time
            )

            eyes_closed = False

            if (
                BLINK_MIN_TIME
                <=
                duration
                <=
                BLINK_MAX_TIME
            ):

                if (
                    now - last_blink_time
                    <=
                    DOUBLE_BLINK_TIME
                ):

                    blink_count += 1

                else:

                    blink_count = 1

                last_blink_time = now

                if (
                    blink_count >= 2
                    and
                    now - last_screenshot_time
                    >
                    SCREENSHOT_COOLDOWN
                ):

                    filename = (
                        "screenshot_"
                        +
                        datetime.now().strftime(
                            "%Y%m%d_%H%M%S"
                        )
                        +
                        ".png"
                    )

                    screenshot = (
                        pyautogui.screenshot()
                    )

                    screenshot.save(
                        filename
                    )

                    last_screenshot_time = now
                    blink_count = 0

                    print(
                        "SCREENSHOT SAVED:",
                        filename
                    )

                    return "SCREENSHOT"

    if (
        blink_count > 0
        and
        now - last_blink_time
        >
        DOUBLE_BLINK_TIME
    ):

        blink_count = 0

    return None


# =========================================================
# EAR GESTURE
# =========================================================

def process_ear_gesture(
    hand,
    face
):

    global left_ear_counter
    global right_ear_counter
    global last_ear_action_time

    index = hand[8]

    left_ear = face[234]
    right_ear = face[454]

    left_distance = math.sqrt(
        (index.x - left_ear.x) ** 2 +
        (index.y - left_ear.y) ** 2
    )

    right_distance = math.sqrt(
        (index.x - right_ear.x) ** 2 +
        (index.y - right_ear.y) ** 2
    )

    now = time.time()

    if left_distance < EAR_DISTANCE:

        left_ear_counter += 1

    else:

        left_ear_counter = 0

    if right_distance < EAR_DISTANCE:

        right_ear_counter += 1

    else:

        right_ear_counter = 0

    if (
        now - last_ear_action_time
        <
        EAR_TAP_COOLDOWN
    ):

        return None

    if (
        left_ear_counter
        >=
        EAR_ACTION_FRAMES
    ):

        pyautogui.press(
            "volumedown"
        )

        left_ear_counter = 0
        right_ear_counter = 0

        last_ear_action_time = now

        print(
            "LEFT EAR - VOLUME DOWN"
        )

        return "VOLUME DOWN"

    if (
        right_ear_counter
        >=
        EAR_ACTION_FRAMES
    ):

        pyautogui.press(
            "volumeup"
        )

        left_ear_counter = 0
        right_ear_counter = 0

        last_ear_action_time = now

        print(
            "RIGHT EAR - VOLUME UP"
        )

        return "VOLUME UP"

    return None


# =========================================================
# DRAW HAND
# =========================================================

def draw_hand(frame, hand):

    h, w, _ = frame.shape

    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (5, 9),
        (9, 13),
        (13, 17)
    ]

    for a, b in connections:

        x1 = int(
            hand[a].x * w
        )

        y1 = int(
            hand[a].y * h
        )

        x2 = int(
            hand[b].x * w
        )

        y2 = int(
            hand[b].y * h
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

    for point in hand:

        x = int(
            point.x * w
        )

        y = int(
            point.y * h
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

if not cap.isOpened():

    print(
        "Camera could not open."
    )

    raise SystemExit


# =========================================================
# START
# =========================================================

print()
print("======================================")
print("              AIRTOUCH")
print("======================================")
print()

print("INDEX              = FAST CURSOR")
print("PINKY UP           = CLICK MODE")
print("1 AIR TAP          = OPEN / DOUBLE CLICK")
print("2 AIR TAPS         = RIGHT CLICK")
print()

print("THUMB UP           = ONE SCROLL UP")
print("THUMB DOWN         = ONE SCROLL DOWN")
print()

print("RIGHT EAR          = VOLUME UP")
print("LEFT EAR           = VOLUME DOWN")
print()

print("BLINK TWICE        = SCREENSHOT")
print()

print("Q                  = QUIT")
print()

print("======================================")
print()


# =========================================================
# MAIN LOOP
# =========================================================

with HandLandmarker.create_from_options(
    hand_options
) as hand_landmarker:

    with FaceLandmarker.create_from_options(
        face_options
    ) as face_landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.flip(
                frame,
                1
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms += 33

            # =================================================
            # HAND DETECTION
            # =================================================

            hand_result = (
                hand_landmarker
                .detect_for_video(
                    image,
                    timestamp_ms
                )
            )

            # =================================================
            # FACE DETECTION
            # =================================================

            face_result = (
                face_landmarker
                .detect_for_video(
                    image,
                    timestamp_ms
                )
            )

            status = "NO HAND"

            # =================================================
            # BLINK
            # =================================================

            if face_result.face_landmarks:

                face = (
                    face_result.face_landmarks[0]
                )

                screenshot_status = (
                    process_blink(
                        face
                    )
                )

                if screenshot_status:

                    status = screenshot_status

            # =================================================
            # HAND FOUND
            # =================================================

            if hand_result.hand_landmarks:

                hand = (
                    hand_result.hand_landmarks[0]
                )

                draw_hand(
                    frame,
                    hand
                )

                # =================================================
                # EAR
                # =================================================

                ear_status = None

                if face_result.face_landmarks:

                    face = (
                        face_result.face_landmarks[0]
                    )

                    ear_status = (
                        process_ear_gesture(
                            hand,
                            face
                        )
                    )

                if ear_status:

                    reset_cursor()
                    reset_scroll()
                    reset_tap()
                    reset_click_sequence()

                    click_mode = False

                    status = ear_status

                else:

                    # =================================================
                    # SCROLL
                    # =================================================

                    scrolling, scroll_status = (
                        process_thumb_scroll(
                            hand
                        )
                    )

                    if scrolling:

                        reset_cursor()
                        reset_tap()
                        reset_click_sequence()

                        click_mode = False

                        status = scroll_status

                    else:

                        reset_scroll()

                        # =================================================
                        # CLICK MODE
                        # =================================================

                        update_click_mode(
                            hand
                        )

                        if click_mode:

                            reset_cursor()

                            process_click(
                                hand
                            )

                            status = "CLICK MODE"

                        else:

                            move_cursor(
                                hand
                            )

                            status = "CURSOR MODE"

            # =================================================
            # NO HAND
            # =================================================

            else:

                reset_cursor()
                reset_scroll()
                reset_tap()
                reset_click_sequence()

                click_mode = False

                pinky_on_counter = 0
                pinky_off_counter = 0

                left_ear_counter = 0
                right_ear_counter = 0

                status = "NO HAND"

            # =================================================
            # UI
            # =================================================

            cv2.putText(
                frame,
                status,
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "INDEX = FAST CURSOR",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "PINKY = CLICK MODE",
                (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "1 TAP = OPEN | 2 TAPS = RIGHT",
                (25, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "THUMB = ONE SCROLL",
                (25, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "EARS = VOLUME",
                (25, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "BLINK x2 = SCREENSHOT",
                (25, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "AirTouch",
                frame
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == ord("q"):

                break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print()
print("AIRTOUCH CLOSED")
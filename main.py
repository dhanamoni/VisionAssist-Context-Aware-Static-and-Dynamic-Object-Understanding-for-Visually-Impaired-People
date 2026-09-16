import cv2
import time
from ultralytics import YOLO
import easyocr

from context import generate_context
from voice import speak


# ==========================================
# LOAD MODELS
# ==========================================

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("Loading OCR model...")

reader = easyocr.Reader(["en"], gpu=False)

print("Models loaded successfully!")


# ==========================================
# OPEN CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found")
    exit()

print("Camera started")


# ==========================================
# TRACKING DATA
# ==========================================

previous_positions = {}

last_spoken = {}

VOICE_COOLDOWN = 3


# ==========================================
# OCR DATA
# ==========================================

last_ocr_time = 0

OCR_INTERVAL = 2

last_ocr_text = ""

last_ocr_spoken = ""

last_ocr_voice_time = 0

OCR_VOICE_COOLDOWN = 5


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = camera.read()

    if not success:
        print("Failed to read camera")
        break


    # ======================================
    # YOLO OBJECT TRACKING
    # ======================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    annotated_frame = result.plot()


    # ======================================
    # OBJECT TRACKING
    # ======================================

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = result.boxes.id.int().cpu().tolist()

        class_ids = result.boxes.cls.int().cpu().tolist()


        # ==================================
        # PROCESS EACH OBJECT
        # ==================================

        for box, track_id, class_id in zip(
            boxes,
            track_ids,
            class_ids
        ):

            # Bounding box
            x1, y1, x2, y2 = map(int, box)


            # Object name
            object_name = model.names[class_id]


            # ==================================
            # OBJECT CENTER
            # ==================================

            center_x = (x1 + x2) // 2

            center_y = (y1 + y2) // 2


            current_position = (
                center_x,
                center_y
            )


            # ==================================
            # MOVEMENT DIRECTION
            # ==================================

            direction = "STATIONARY"


            if track_id in previous_positions:

                previous_x, previous_y = previous_positions[track_id]

                dx = center_x - previous_x

                dy = center_y - previous_y

                threshold = 10


                if abs(dx) > threshold:

                    if dx > 0:
                        direction = "RIGHT"

                    else:
                        direction = "LEFT"


                elif abs(dy) > threshold:

                    if dy > 0:
                        direction = "DOWN"

                    else:
                        direction = "UP"


            # Save current position

            previous_positions[track_id] = current_position


            # ==================================
            # DISTANCE ESTIMATION
            # ==================================

            box_width = x2 - x1

            box_height = y2 - y1

            box_area = box_width * box_height


            if box_area > 150000:

                distance = "VERY NEAR"

            elif box_area > 70000:

                distance = "NEAR"

            elif box_area > 25000:

                distance = "MEDIUM"

            else:

                distance = "FAR"


            # ==================================
            # CONTEXT ENGINE
            # ==================================

            context_message = generate_context(
                object_name,
                direction,
                distance,
                center_x,
                frame.shape[1]
            )


            # ==================================
            # DISPLAY OBJECT NAME
            # ==================================

            cv2.putText(
                annotated_frame,
                f"{object_name} | ID {track_id}",
                (
                    x1,
                    max(y1 - 50, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # ==================================
            # DISPLAY DIRECTION
            # ==================================

            cv2.putText(
                annotated_frame,
                f"Direction: {direction}",
                (
                    x1,
                    max(y1 - 30, 40)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # ==================================
            # DISPLAY DISTANCE
            # ==================================

            cv2.putText(
                annotated_frame,
                f"Distance: {distance}",
                (
                    x1,
                    max(y1 - 10, 60)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )


            # ==================================
            # VOICE WARNING
            # ==================================

            current_voice_time = time.time()

            last_time = last_spoken.get(
                track_id,
                0
            )


            if (
                distance == "VERY NEAR"
                and
                current_voice_time - last_time
                > VOICE_COOLDOWN
            ):

                speak(context_message)

                last_spoken[track_id] = current_voice_time


    # ======================================
    # OCR EVERY 2 SECONDS
    # ======================================

    current_time = time.time()


    if (
        current_time - last_ocr_time
        >= OCR_INTERVAL
    ):

        last_ocr_time = current_time

        print("Running OCR...")


        # Run OCR
        ocr_results = reader.readtext(frame)


        # ==================================
        # PROCESS OCR RESULTS
        # ==================================

        for bbox, text, confidence in ocr_results:

            # Ignore low-confidence text

            if confidence < 0.50:
                continue


            # ==================================
            # SAVE OCR TEXT
            # ==================================

            last_ocr_text = text


            # ==================================
            # OCR VOICE
            # ==================================

            current_ocr_time = time.time()


            if (
                text != last_ocr_spoken
                or
                current_ocr_time - last_ocr_voice_time
                > OCR_VOICE_COOLDOWN
            ):

                speak(
                    f"Text detected: {text}"
                )

                last_ocr_spoken = text

                last_ocr_voice_time = current_ocr_time


            # ==================================
            # OCR BOUNDING BOX
            # ==================================

            top_left = tuple(
                map(int, bbox[0])
            )

            bottom_right = tuple(
                map(int, bbox[2])
            )


            cv2.rectangle(
                annotated_frame,
                top_left,
                bottom_right,
                (255, 0, 0),
                2
            )


            # ==================================
            # DISPLAY OCR TEXT
            # ==================================

            cv2.putText(
                annotated_frame,
                f"{text} ({confidence:.2f})",
                (
                    top_left[0],
                    max(
                        top_left[1] - 10,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )


            # Terminal output

            print(
                f"OCR: {text} | "
                f"Confidence: {confidence:.2f}"
            )


    # ======================================
    # DISPLAY LAST OCR TEXT
    # ======================================

    if last_ocr_text:

        cv2.putText(
            annotated_frame,
            f"Text: {last_ocr_text}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )


    # ======================================
    # APPLICATION TITLE
    # ======================================

    cv2.putText(
        annotated_frame,
        "VisionAssist - Integrated System",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ======================================
    # INSTRUCTIONS
    # ======================================

    cv2.putText(
        annotated_frame,
        "Press Q or ESC to exit",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "VisionAssist",
        annotated_frame
    )


    # ======================================
    # KEYBOARD INPUT
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q") or key == 27:

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()

print("VisionAssist stopped.")
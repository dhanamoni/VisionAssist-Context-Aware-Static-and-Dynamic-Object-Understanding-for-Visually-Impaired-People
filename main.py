
import cv2
import time
import numpy as np
import easyocr

from ultralytics import YOLO
from context import generate_context
from voice import speak


# ==========================================
# 1. LOAD MODELS
# ==========================================

print("Loading YOLO model...")

model = YOLO("models/yolo11n.pt")

print("Loading OCR model...")

reader = easyocr.Reader(["en"], gpu=False)

print("Models loaded successfully!")


# ==========================================
# 2. SETTINGS
# ==========================================

OCR_INTERVAL = 2
OBJECT_VOICE_INTERVAL = 6
TEXT_REPEAT_COOLDOWN = 8

OCR_CONFIDENCE = 0.30
OBJECT_CONFIDENCE = 0.40

last_ocr_time = 0
last_object_voice_time = 0

# Stores the last time each text was spoken.
last_spoken_text = {}

# Stores previous object positions.
previous_positions = {}

# Stores OCR results for displaying.
ocr_results = []


# ==========================================
# 3. OCR IMAGE PREPROCESSING
# ==========================================

def preprocess_image(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.convertScaleAbs(
        gray,
        alpha=1.5,
        beta=10
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    height, width = gray.shape

    gray = cv2.resize(
        gray,
        (width * 2, height * 2),
        interpolation=cv2.INTER_CUBIC
    )

    return gray


# ==========================================
# 4. ESTIMATE OBJECT DISTANCE
# ==========================================

def estimate_distance(box_height, frame_height):

    ratio = box_height / frame_height

    if ratio > 0.50:
        return "very near"

    elif ratio > 0.25:
        return "near"

    else:
        return "far away"


# ==========================================
# 5. ESTIMATE OBJECT MOVEMENT
# ==========================================

def estimate_direction(track_id, center_x):

    if track_id is None:
        return "stationary"

    if track_id not in previous_positions:

        previous_positions[track_id] = center_x

        return "stationary"

    old_x = previous_positions[track_id]

    difference = center_x - old_x

    previous_positions[track_id] = center_x

    if difference > 15:
        return "moving right"

    elif difference < -15:
        return "moving left"

    return "stationary"


# ==========================================
# 6. START WEBCAM
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    raise SystemExit


print("Camera started.")
print("VisionAssist is running.")
print("Press Q to stop.")


# ==========================================
# 7. MAIN PROGRAM
# ==========================================

try:

    while True:

        success, frame = camera.read()

        if not success:

            print("Camera frame unavailable.")

            break

        current_time = time.time()

        frame_height, frame_width = frame.shape[:2]

        ocr_spoken_this_frame = False

        # IMPORTANT:
        # Keep an untouched copy of the camera frame.
        # OCR will only process this clean copy.

        clean_frame = frame.copy()


        # ==================================
        # 8. OCR ON CLEAN CAMERA FRAME
        # ==================================

        if current_time - last_ocr_time >= OCR_INTERVAL:

            processed_frame = preprocess_image(clean_frame)

            detected_results = reader.readtext(
                processed_frame,
                detail=1,
                paragraph=False,
                decoder="greedy",
                text_threshold=0.5,
                low_text=0.3,
                link_threshold=0.3
            )

            print("\nOCR raw results:", detected_results)

            current_texts = set()

            ocr_results = []

            for result in detected_results:

                bbox, text, confidence = result

                text = " ".join(text.strip().split())

                if confidence < OCR_CONFIDENCE:
                    continue

                if len(text) < 3:
                    continue

                if not any(char.isalpha() for char in text):
                    continue

                current_texts.add(text)

                ocr_results.append(
                    (bbox, text, confidence)
                )

                print(
                    "OCR detected:",
                    text,
                    "Confidence:",
                    round(float(confidence), 2)
                )


            # ==================================
            # 9. SPEAK NEW OR CHANGED TEXT
            # ==================================

            texts_to_speak = []

            speech_time = time.time()

            for text in sorted(current_texts):

                text_key = text.casefold()

                previous_time = last_spoken_text.get(
                    text_key,
                    0
                )

                if (
                    speech_time - previous_time
                    >= TEXT_REPEAT_COOLDOWN
                ):

                    texts_to_speak.append(text)


            if texts_to_speak:

                message = (
                    "I can read: "
                    + ". ".join(texts_to_speak)
                )

                print("\nABOUT TO SPEAK OCR:", message)

                try:

                    speak(message)

                    print("OCR speech finished.")

                    spoken_time = time.time()

                    for text in texts_to_speak:

                        last_spoken_text[
                            text.casefold()
                        ] = spoken_time

                    ocr_spoken_this_frame = True

                except Exception as error:

                    print("OCR voice error:", error)


            last_ocr_time = time.time()


        # ==================================
        # 10. YOLO OBJECT DETECTION
        # ==================================

        # YOLO receives the original frame.
        # Its labels are drawn only after OCR.

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detected_objects = []

        if results and results[0].boxes is not None:

            for box in results[0].boxes:

                confidence = float(box.conf[0])

                if confidence < OBJECT_CONFIDENCE:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                class_id = int(box.cls[0])

                object_name = model.names[class_id]

                box_height = max(1, y2 - y1)

                center_x = (x1 + x2) // 2

                distance = estimate_distance(
                    box_height,
                    frame_height
                )

                track_id = None

                if box.id is not None:

                    track_id = int(box.id[0])

                direction = estimate_direction(
                    track_id,
                    center_x
                )

                context_message = generate_context(
                    object_name,
                    direction,
                    distance,
                    center_x,
                    frame_width
                )

                detected_objects.append({
                    "name": object_name,
                    "message": context_message,
                    "height": box_height
                })


                # Draw object rectangle.

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"{object_name} {confidence:.2f}"
                )

                if track_id is not None:

                    label += f" ID:{track_id}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


        # ==================================
        # 11. DRAW OCR TEXT
        # ==================================

        # OCR coordinates are based on the
        # image resized to twice its size.
        # Divide coordinates by 2.

        for bbox, text, confidence in ocr_results:

            points = []

            for point in bbox:

                original_x = int(point[0] / 2)

                original_y = int(point[1] / 2)

                points.append([
                    original_x,
                    original_y
                ])

            points_array = np.array(
                points,
                dtype=np.int32
            )

            cv2.polylines(
                frame,
                [points_array],
                True,
                (255, 0, 0),
                2
            )

            text_x = points[0][0]

            text_y = max(
                20,
                points[0][1] - 10
            )

            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )


        # ==================================
        # 12. OBJECT VOICE WARNING
        # ==================================

        # Do not speak object warnings while
        # OCR is announcing recognized text.

        if detected_objects and not ocr_spoken_this_frame:

            current_time = time.time()

            if (
                current_time - last_object_voice_time
                >= OBJECT_VOICE_INTERVAL
            ):

                # Select the object with the
                # largest bounding-box height.

                closest_object = max(
                    detected_objects,
                    key=lambda item: item["height"]
                )

                context_message = closest_object["message"]

                print(
                    "\nObject context:",
                    context_message
                )

                try:

                    speak(context_message)

                    last_object_voice_time = time.time()

                except Exception as error:

                    print("Object voice error:", error)


        # ==================================
        # 13. DISPLAY CAMERA
        # ==================================

        cv2.imshow(
            "VisionAssist - Object Detection and OCR",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            print("Stopping VisionAssist...")

            break


# ==========================================
# 14. RELEASE CAMERA
# ==========================================

finally:

    camera.release()

    cv2.destroyAllWindows()

    print("VisionAssist stopped.")

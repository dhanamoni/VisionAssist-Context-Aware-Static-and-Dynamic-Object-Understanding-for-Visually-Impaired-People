import cv2
import easyocr

# -----------------------------
# CREATE OCR READER
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)

# -----------------------------
# OPEN CAMERA
# -----------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found")
    exit()

print("Camera started")
print("Show clear text in front of the camera")
print("Press Q to quit")

while True:

    success, frame = camera.read()

    if not success:
        print("Failed to read camera")
        break

    # -----------------------------
    # OCR
    # -----------------------------
    results = reader.readtext(frame)

    # -----------------------------
    # DISPLAY OCR RESULTS
    # -----------------------------
    for bbox, text, confidence in results:

        # Convert coordinates
        top_left = tuple(map(int, bbox[0]))
        top_right = tuple(map(int, bbox[1]))
        bottom_right = tuple(map(int, bbox[2]))
        bottom_left = tuple(map(int, bbox[3]))

        # Draw box
        cv2.line(
            frame,
            top_left,
            top_right,
            (0, 255, 0),
            2
        )

        cv2.line(
            frame,
            top_right,
            bottom_right,
            (0, 255, 0),
            2
        )

        cv2.line(
            frame,
            bottom_right,
            bottom_left,
            (0, 255, 0),
            2
        )

        cv2.line(
            frame,
            bottom_left,
            top_left,
            (0, 255, 0),
            2
        )

        # Display detected text
        display_text = f"{text} ({confidence:.2f})"

        cv2.putText(
            frame,
            display_text,
            (top_left[0], max(top_left[1] - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Print text in terminal
        print(
            f"Detected: {text} | Confidence: {confidence:.2f}"
        )

    # -----------------------------
    # TITLE
    # -----------------------------
    cv2.putText(
        frame,
        "VisionAssist - OCR",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # SHOW CAMERA
    # -----------------------------
    cv2.imshow(
        "VisionAssist - OCR",
        frame
    )

    # -----------------------------
    # QUIT
    # -----------------------------
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
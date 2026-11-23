
import cv2
import easyocr
from voice import speak

print("Loading OCR...")
reader = easyocr.Reader(['en'], gpu=False)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found")
    exit()

print("Camera started. Show printed English text.")
print("Press Q to quit.")

last_text = ""

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        results = reader.readtext(frame)

        for bbox, text, confidence in results:

            if confidence >= 0.40:

                text = text.strip()

                if text:
                    cv2.putText(
                        frame,
                        text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    if text != last_text:
                        print("Detected:", text)
                        speak("Text detected: " + text)
                        last_text = text

        cv2.imshow("VisionAssist OCR Voice Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("OCR Voice Test Stopped.")

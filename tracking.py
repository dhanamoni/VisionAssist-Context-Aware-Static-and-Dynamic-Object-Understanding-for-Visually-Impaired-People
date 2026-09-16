from ultralytics import YOLO
import cv2

# Import Context Engine
from context import generate_context


# -----------------------------------
# LOAD YOLO MODEL
# -----------------------------------

model = YOLO("yolo11n.pt")


# -----------------------------------
# OPEN WEBCAM
# -----------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found")
    exit()


# -----------------------------------
# STORE PREVIOUS OBJECT POSITIONS
# -----------------------------------

previous_positions = {}


# -----------------------------------
# MAIN LOOP
# -----------------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("Failed to read camera")
        break


    # -----------------------------------
    # YOLO OBJECT TRACKING
    # -----------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    result = results[0]


    # -----------------------------------
    # DRAW YOLO RESULTS
    # -----------------------------------

    annotated_frame = result.plot()


    # -----------------------------------
    # CHECK TRACKING RESULTS
    # -----------------------------------

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = result.boxes.id.int().cpu().tolist()

        class_ids = result.boxes.cls.int().cpu().tolist()


        # -----------------------------------
        # PROCESS EACH OBJECT
        # -----------------------------------

        for box, track_id, class_id in zip(
            boxes,
            track_ids,
            class_ids
        ):

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box)


            # -----------------------------------
            # OBJECT NAME
            # -----------------------------------

            object_name = model.names[class_id]


            # -----------------------------------
            # OBJECT CENTER
            # -----------------------------------

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            current_position = (
                center_x,
                center_y
            )


            # -----------------------------------
            # MOVEMENT DIRECTION
            # -----------------------------------

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


            # -----------------------------------
            # DISTANCE ESTIMATION
            # -----------------------------------

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


            # -----------------------------------
            # CONTEXT ENGINE
            # -----------------------------------

            context_message = generate_context(
                object_name,
                direction,
                distance,
                center_x,
                frame.shape[1]
            )


            # -----------------------------------
            # DISPLAY OBJECT INFORMATION
            # -----------------------------------

            text1 = f"{object_name} | ID {track_id}"

            text2 = f"Direction: {direction}"

            text3 = f"Distance: {distance}"


            cv2.putText(
                annotated_frame,
                text1,
                (x1, y1 - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            cv2.putText(
                annotated_frame,
                text2,
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            cv2.putText(
                annotated_frame,
                text3,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )


            # -----------------------------------
            # DISPLAY CONTEXT MESSAGE
            # -----------------------------------

            cv2.putText(
                annotated_frame,
                context_message,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


    # -----------------------------------
    # SHOW CAMERA
    # -----------------------------------

    cv2.imshow(
        "VisionAssist - Context Engine",
        annotated_frame
    )


    # -----------------------------------
    # PRESS Q TO EXIT
    # -----------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------------
# RELEASE RESOURCES
# -----------------------------------

camera.release()

cv2.destroyAllWindows()
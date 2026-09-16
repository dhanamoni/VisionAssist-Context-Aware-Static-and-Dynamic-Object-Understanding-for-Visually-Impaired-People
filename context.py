def generate_context(object_name, direction, distance, center_x, frame_width):

    # Determine which side of the camera the object is on
    if center_x < frame_width * 0.33:
        side = "left"

    elif center_x > frame_width * 0.66:
        side = "right"

    else:
        side = "center"

    # Very near object
    if distance == "VERY NEAR":
        return f"Warning! {object_name} is very near on your {side}."

    # Near object
    elif distance == "NEAR":

        if direction != "STATIONARY":
            return (
                f"{object_name} is near on your {side}, "
                f"moving {direction.lower()}."
            )

        return f"{object_name} is near on your {side}."

    # Medium distance
    elif distance == "MEDIUM":
        return f"{object_name} is at medium distance on your {side}."

    # Far object
    else:
        return f"{object_name} is far away on your {side}."
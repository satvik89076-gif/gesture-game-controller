import cv2

# Initialize webcam (0 is usually the default built-in camera)
cap = cv2.VideoCapture(0)

print("Starting camera feed... Press 'q' on your keyboard to exit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read from camera. Check camera permissions.")
        break

    # Flip horizontally for a mirror effect (feels natural for gaming)
    frame = cv2.flip(frame, 1)

    # Display status on the video feed
    cv2.putText(
        frame,
        "Webcam Test: Working! Press 'q' to exit",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # Show the live window
    cv2.imshow("Hand Controller - Camera Test", frame)

    # Press 'q' to quit the window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close window
cap.release()
cv2.destroyAllWindows()

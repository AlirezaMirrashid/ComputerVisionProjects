import cv2
import numpy as np

def motion_tracking(video_path, output_path=None):
    """
    Tracks motion in a video by detecting moving objects using background subtraction.

    Parameters:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the processed video (optional).
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

    # Output video writer setup
    writer = None
    if output_path:

        # Output video writer setup (MP4 format)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is for MP4 files
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fg_mask = back_sub.apply(frame)

        # Threshold the mask to remove noise
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours of the detected motion
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                # Get bounding box for each moving object
                x, y, w, h = cv2.boundingRect(contour)

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Motion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the processed frame
        cv2.imshow("Motion Tracking", frame)

        # Save to output video
        if writer:
            writer.write(frame)

        # Break on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

# Example usage:
video_path = "samples/video.mp4"  # Replace with your video path
output_path = "output_motion_tracking.avi"  # Replace with desired output path or None
motion_tracking(video_path, output_path)

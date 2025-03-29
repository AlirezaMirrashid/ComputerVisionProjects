import cv2
import threading
import time
import queue


# Multi-threaded class for video reading
class CapMultiThreading:
    def __init__(self, cap_id):
        assert cap_id is not None, "Please insert a camera id (Integer or String)"

        self.frame_count = 0
        self.frame_queue = queue.Queue(maxsize=256)
        self.cap = None
        self.cap_id = cap_id
        self.ended = False

        # Start the frame reading thread
        self.start()

    def start(self):
        self.cap = cv2.VideoCapture(self.cap_id)
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.ended = True
                    break
                self.frame_queue.put([ret, frame])
                self.frame_count += 1
            # else:
            #     pass

    def read(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None

    def has_ended(self):
        return self.ended

    def release(self):
        self.cap.release()


# Single-threaded function for reading and displaying frames
def single_threaded_read(video_path):
    cap = cv2.VideoCapture(video_path)
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Display FPS on top-left corner of the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('Frame', frame)

        # Wait for a brief moment before showing the next frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    print(f"Single-threaded FPS: {fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()


# Multi-threaded function for reading and displaying frames
def multi_threaded_read(video_path):
    video_stream = CapMultiThreading(video_path)
    start_time = time.time()
    frame_count = 0

    while not video_stream.has_ended():
        frame_data = video_stream.read()

        if frame_data is not None:
            ret, frame = frame_data
            if ret:
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time

                # Display FPS on top-left corner of the frame
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display frame
                cv2.imshow('Frame', frame)

        # Wait for a brief moment before showing the next frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    print(f"Multi-threaded FPS: {fps:.2f}")

    video_stream.release()
    cv2.destroyAllWindows()


# Main function to run both single-threaded and multi-threaded tests
def main():
    video_path = 'flow.mp4'  # Replace with your video path

    print("Testing single-threaded FPS...")
    single_threaded_read(video_path)

    print("Testing multi-threaded FPS...")
    multi_threaded_read(video_path)


if __name__ == "__main__":
    main()
import os
from time import sleep
import cv2
import numpy as np
from webcam import open_webcam_video
import dotenv


def normalize_image(image):
    # Normalize the image so that the darkest pixel becomes 0 and the brightest becomes 255
    norm_image = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    return norm_image


def test_normalize():
    dotenv.load_dotenv(override=True)
    WEBCAM_INDEX = os.getenv("WEBCAM_INDEX")
    cap = open_webcam_video(WEBCAM_INDEX)

    while True:
        sleep(0.5)
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Failed to grab frame")
            break

        # Normalize the frame
        normalized_frame = normalize_image(frame)

        # Add labels to each frame
        cv2.putText(
            frame,
            "Original",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            normalized_frame,
            "Normalized",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Stack the original and normalized frames vertically
        stacked_frames = np.hstack((frame, normalized_frame))

        # Display the stacked frames
        cv2.imshow("Webcam - Original and Normalized", stacked_frames)

        # Press 'q' to close the video window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_normalize()

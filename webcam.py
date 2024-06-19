import platform
import traceback
import cv2


def open_webcam_video(webcam_index):
    video = None
    if platform.system() == "Windows":
        if webcam_index.isdigit():
            video = cv2.VideoCapture(int(webcam_index), cv2.CAP_DSHOW)
        else:
            video = cv2.VideoCapture(webcam_index)

    # darwin is mac
    elif platform.system() == "Linux" or platform.system() == "Darwin":
        if webcam_index.isdigit():
            webcam_index = int(webcam_index)
        video = cv2.VideoCapture(webcam_index)
    else:
        traceback.print_exc()
        raise Exception("Unsupported platform. Please use Windows, Linux or Mac.")
    return video

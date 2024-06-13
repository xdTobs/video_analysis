import cv2
import os
import datetime
import numpy as np


class VideoDebugger:
    def __init__(self) -> None:
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.videos = {}

    def write_video(self, out_name: str, frame: np.ndarray, three_channel=False):
        if out_name not in self.videos:
            os.makedirs(f"output/{out_name}", exist_ok=True)
            video = cv2.VideoWriter(
                f"output/{out_name}/{self.current_time}.avi",
                self.fourcc,
                10.0,
                (1024, 576),
            )
            self.videos[out_name] = video

        video = self.videos[out_name]
        if not three_channel:
            res = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            res = frame

        video.write(res)
        return frame

    def run_analysis(
        self, anal_func, out_name, frame, bounds_dict_entry, three_channel=False
    ):
        mask = anal_func(frame, bounds_dict_entry)
        self.write_video(out_name, mask, three_channel)
        return mask

    def close_videos(
        self,
    ):
        for video in self.videos.values():
            video.release()
        self.videos.clear()

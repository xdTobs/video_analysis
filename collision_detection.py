import cv2
import numpy as np
from analyse import Analyse


def collision_detection(image: np.ndarray):
    print("Collision detection test for image " + img_path)
    print(img)
    analyser = Analyse()
    analyser.analysis_pipeline(img)
    print(analyser)


if __name__ == "__main__":
    img_path = "./videos/out.jpg"
    img = cv2.imread(img_path)
    collision_detection(img)

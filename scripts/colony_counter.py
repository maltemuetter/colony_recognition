import cv2
import numpy as np
from .colony_recognition import ColonyRecognition
import pandas as pd

ABSORBANCE_CONSTANT = 0.8
MAX_PIXEL_VALUE = 255


class ColonyCounter:
    def __init__(self, background_path, x_start=50, y_start=150, x_end=1550, y_end=1050, number_of_sections=6, horizontal=True):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.horizontal = horizontal
        self.number_of_sections = number_of_sections
        self.background = cv2.imread(background_path)[
            y_start:y_end, x_start:x_end]

    def analyze_section(self, img_path, sec_number):
        image = cv2.imread(img_path)[
            self.y_start:self.y_end, self.x_start:self.x_end]
        back_sec = get_sec(sec_number, self.background)
        im_sec = get_sec(sec_number, image)
        return ColonyRecognition(absorbance_image(im_sec, back_sec))


def absorbance_image(image, background):
    """Return the absorbance of the image relative to the background."""
    gray_image = to_positive_gray(image)
    gray_background = to_positive_gray(background)
    absorbance = np.log(gray_background / gray_image)
    scaled = absorbance * MAX_PIXEL_VALUE / np.log(1 / ABSORBANCE_CONSTANT)
    return np.clip(scaled, 0, MAX_PIXEL_VALUE).astype(np.uint8)


def to_positive_gray(image):
    """Convert a BGR image to grayscale floats of at least one."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    return np.maximum(gray, 1)


def get_sec(sec, img):
    shape = img.shape
    w = shape[1]
    dim = len(shape)
    d = w/8

    reps = [1, 2, 3, 4]

    section_infos = []
    for section in range(8):
        section_infos.append({
            "section": section,
            "x_start": round(section*d),
            "x_end": round((section+1)*d),
            "r": reps[section % 4],
            "exp": ["exp_1", "exp_2"][section >= 4]
        })
    df = pd.DataFrame(section_infos).set_index("section")

    if dim == 2:
        frame = img[:, df.loc[sec, "x_start"]:df.loc[sec, "x_end"]]
    elif dim == 3:
        frame = img[:, df.loc[sec, "x_start"]:df.loc[sec, "x_end"]]

    return frame

import itertools
import time

import cv2


def show_and_wait(img, name="img"):
    cv2.imshow(name, img)
    while cv2.waitKey() != ord('q'):
        continue
    cv2.destroyAllWindows()


def check_if_generator_is_empty(generator):
    try:
        first = next(generator)
    except StopIteration:
        return None
    return itertools.chain([first], generator)

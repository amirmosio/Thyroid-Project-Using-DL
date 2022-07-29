import itertools

import cv2


def show_and_wait(img, name="img", wait=True, save=False):
    cv2.imshow(name, img)
    if wait:
        while cv2.waitKey() != ord('q'):
            continue
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite(f"{name}.jpeg", img)


def check_if_generator_is_empty(generator):
    try:
        first = next(generator)
    except StopIteration:
        return None
    return itertools.chain([first], generator)

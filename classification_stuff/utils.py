import time

import cv2


def show_and_wait(img, name="img"):
    cv2.imshow(name, img)
    while cv2.waitKey() != ord('q'):
        continue
    cv2.destroyAllWindows()

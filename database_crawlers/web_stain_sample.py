import enum
import json
import time
from io import BytesIO
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image
from tifffile import TiffWriter

from database_crawlers.utils import find_in_log_n, fetch_tile_content, download_urls_in_thread


class StainType(enum.Enum):
    H_AND_E = 0, "H&E"
    UNKNOWN = 1, "UNKNOWN"


class ThyroidType(enum.Enum):
    UNKNOWN = -1, "UNKNOWN"
    NORMAL = 0, "NORMAL"
    PAPILLARY_CARCINOMA = 1, "PAPILLARY_CARCINOMA"
    GRAVES_DISEASE = 2, "GRAVES_DISEASE"
    NODULAR_GOITER = 3, "NODULAR_GOITER"
    HASHIMOTO_THYROIDITIS = 4, "HASHIMOTO_THYROIDITIS"
    FOLLICULAR_CARCINOMA = 5, "FOLLICULAR_CARCINOMA"
    FOLLICULAR_ADENOMA = 6, "FOLLICULAR_ADENOMA"
    COLLOID_GOITER = 7, "COLLOID_GOITER"

    @staticmethod
    def get_thyroid_type_from_diagnosis_label(label: str):
        label = label.lower()
        if "normal" in label:
            return ThyroidType.NORMAL
        elif "papillary" in label:
            return ThyroidType.PAPILLARY_CARCINOMA
        elif "grave" in label:
            return ThyroidType.GRAVES_DISEASE
        elif "nodular" in label and "goiter" in label:
            return ThyroidType.NODULAR_GOITER
        elif "hashimoto" in label:
            return ThyroidType.HASHIMOTO_THYROIDITIS
        elif "follicular" in label:
            if "adenoma" in label:
                return ThyroidType.FOLLICULAR_ADENOMA
            else:
                return ThyroidType.FOLLICULAR_CARCINOMA
        elif "colloid" in label and "goiter" in label:
            return ThyroidType.COLLOID_GOITER
        else:
            return ThyroidType.UNKNOWN


class WebStainImage:
    save_path = "data/"

    def __init__(self, database_name, image_id, image_web_label, report, stain_type, is_wsi):
        self.database_name = database_name
        self.image_id = image_id
        self.image_web_label = image_web_label
        self.report = report
        self.stain_type = stain_type
        self.is_wsi = is_wsi

    def to_json(self):
        return {"database_name": self.database_name,
                "image_id": self.image_id,
                "image_web_label": self.image_web_label,
                "image_class_label": self.image_class_label,
                "report": self.report,
                "stain_type": self.stain_type.value[1],
                "is_wsi": self.is_wsi}

    @staticmethod
    def sorted_json_keys():
        return ["database_name",
                "image_id",
                "image_web_label",
                "image_class_label",
                "report",
                "stain_type",
                "is_wsi"]

    @property
    def image_class_label(self):
        return ThyroidType.get_thyroid_type_from_diagnosis_label(self.image_web_label).value[1]

    def get_slide_view_url(self):
        raise NotImplemented("get_slide_view_url")

    def crawl_image_save_jpeg_and_json(self):
        raise NotImplemented("crawl_image_get_jpeg")

    def _get_file_path_name(self):
        return self.save_path + self.image_id

    def _get_relative_image_path(self):
        return self._get_file_path_name() + ".jpeg"

    def _get_relative_tiff_image_path(self):
        return self._get_file_path_name() + ".tiff"

    def _get_relative_json_path(self):
        return self._get_file_path_name() + ".json"

    def _save_json_file(self):
        json_object = json.dumps(self.to_json())
        with open(self._get_relative_json_path(), "w") as outfile:
            outfile.write(json_object)


class WebStainWSI(WebStainImage):
    def __init__(self, database_name, image_id, image_web_label, report, stain_type, is_wsi):
        super().__init__(database_name, image_id, image_web_label, report, stain_type, is_wsi)

    def _get_tile_url(self, zoom, partition=None, i=None, j=None):
        raise NotImplemented("_get_tile_url")

    def _generate_tile_urls(self):
        raise NotImplemented("generate tile urls")

    def find_best_zoom(self):
        return 0

    def _find_first_tile_width(self):
        image_content = fetch_tile_content(self._get_tile_url(self.find_best_zoom(), partition=0, i=0, j=0))
        img = Image.open(BytesIO(image_content))
        return img.size[0], img.size[1]

    def _fetch_all_tiles(self):
        batch = []
        index = 0
        for url in self._generate_tile_urls():
            batch.append((url, index))
            # DONE
            index += 1
        # download last batch
        if len(batch) != 0:
            for content, downloaded_index in download_urls_in_thread(batch):
                yield content, downloaded_index
        print("Slide download tiles done!!!")

    def crawl_image_save_jpeg_and_json(self):
        raise NotImplemented("crawl_image_save_jpeg_and_json")


class WebStainWSIOneDIndex(WebStainWSI):
    def __init__(self, database_name, image_id, image_web_label, report, stain_type, is_wsi):
        super().__init__(database_name, image_id, image_web_label, report, stain_type, is_wsi)
        self.last_partition = None

    def _find_last_partition(self):
        print("Finding last partition: ", end="")

        def func(partition, retry=3):
            print(partition, end="")
            for i in range(retry):
                try:
                    request = Request(self._get_tile_url(self.find_best_zoom(), partition=partition), method='HEAD')
                    resp = urlopen(request)
                    headers = resp.info()
                    print("<", end=", ")
                    return True
                except Exception as e:
                    print("e", end="")
                    time.sleep(2 ** (0.1 * (i + 1)))
            print(">", end=", ")
            return False

        return find_in_log_n(0, 1000 * 1000, func)

    def _generate_tile_urls(self):
        for partition in range(self.last_partition + 1):
            yield self._get_tile_url(self.find_best_zoom(), partition=partition)

    def crawl_image_save_jpeg_and_json(self):
        def generator():
            while True:
                if first_temp_rows:
                    yield first_temp_rows[0]
                    del first_temp_rows[0]
                else:
                    res = next(content_fetcher, -1)
                    if res == -1:
                        break
                    img = cv2.imdecode(np.frombuffer(res[0], np.uint8), -1)
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    yield img

        first_image_width, first_image_height = self._find_first_tile_width()
        first_temp_rows = []
        column_tiles, row_tiles = None, None
        self.last_partition = self._find_last_partition()
        content_fetcher = self._fetch_all_tiles()
        with TiffWriter(self._get_relative_tiff_image_path(), bigtiff=True) as tif:
            while column_tiles is None:
                content, index = content_fetcher.__next__()
                image_array = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                first_temp_rows.append(image_array)
                if image_array.shape[1] != first_image_width:
                    column_tiles = index + 1
                    row_tiles = (self.last_partition + 1) // column_tiles
            shape = (first_image_height * row_tiles, first_image_width * column_tiles, 3)
            tif.write(generator(), subfiletype=1, tile=(first_image_height, first_image_width), shape=shape,
                      dtype=np.uint8,
                      compression='JPEG',  # TODO
                      photometric='rgb')

        """
        Save json file
        """
        self._save_json_file()


class WebStainWSITwoDIndex(WebStainWSI):
    def __init__(self, database_name, image_id, image_web_label, report, stain_type, is_wsi):
        super().__init__(database_name, image_id, image_web_label, report, stain_type, is_wsi)
        self.last_i = None
        self.last_j = None

    def _generate_tile_urls(self):
        for j in range(self.last_j + 1):
            for i in range(self.last_i + 1):
                yield self._get_tile_url(self.find_best_zoom(), i=i, j=j)

    def _find_last_i_and_j(self):
        def func(i, j, retry=3):
            print(f"{i}-{j}", end="")
            for r in range(retry):
                try:
                    request = Request(self._get_tile_url(self.find_best_zoom(), i=i, j=j), method='HEAD')
                    resp = urlopen(request)
                    headers = resp.info()
                    print("<", end=", ")
                    return True
                except Exception as e:
                    print("e", end="")
                    time.sleep(2 ** (0.1 * (r + 1)))
            print(">", end=", ")
            return False

        print("Finding last i: ", end="")
        i_func = lambda i: func(i=i, j=0)
        last_i = find_in_log_n(0, 1000, i_func)
        print("\nFinding last j: ")
        j_func = lambda j: func(i=0, j=j)
        last_j = find_in_log_n(0, 1000, j_func)
        return last_i, last_j

    def crawl_image_save_jpeg_and_json(self):
        def generator():
            while True:
                res = next(content_fetcher, -1)
                if res == -1:
                    break
                res = cv2.imdecode(np.frombuffer(res[0], np.uint8), -1)
                if max(res.shape) >= 260:
                    raise Exception(f"warning shape: {res.shape}")
                res = cv2.resize(res, (min(res.shape[1], 256), min(res.shape[0], 256)))
                yield res

        first_image_width = 256
        first_image_height = 256
        self.last_i, self.last_j = self._find_last_i_and_j()
        content_fetcher = self._fetch_all_tiles()
        with TiffWriter(self._get_relative_tiff_image_path(), bigtiff=True) as tif:
            shape = (first_image_height * (self.last_j + 1), first_image_width * (self.last_i + 1), 3)
            tif.write(generator(), subfiletype=1,
                      tile=(first_image_height, first_image_width),
                      shape=shape,
                      dtype=np.uint8,
                      compression='JPEG',  # TODO
                      photometric='rgb')

        """
        Save json file
        """
        self._save_json_file()

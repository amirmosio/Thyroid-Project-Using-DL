from math import ceil

import cv2

from data_loader import zarr_loader


class ThyroidFragmentFilters:
    @staticmethod
    def empty_frag_with_laplacian(image_nd_array, threshold=400):
        gray = cv2.cvtColor(image_nd_array, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3, )
        std = cv2.cv2.meanStdDev(laplacian)[1][0][0]

        variance = std ** 2

        return variance >= threshold


def generate_raw_fragments_from_zarr(zarr_object, frag_size=512, frag_overlap=0.1):
    zarr_shape = zarr_object.shape

    step_size = int(frag_size * (1 - frag_overlap))
    overlap_size = frag_size - step_size
    for w in range(0, ceil((zarr_shape[0] - overlap_size) / step_size) * step_size, step_size):
        for h in range(0, ceil((zarr_shape[1] - overlap_size) / step_size) * step_size, step_size):
            end_w, end_h = min(zarr_shape[0], w + frag_size), min(zarr_shape[1], h + frag_size)
            start_w, start_h = end_w - frag_size, end_h - frag_size
            yield zarr_object[start_w:end_w, start_h: end_h]


def filter_frag_from_generator(frag_generator, filter_func_list):
    for frag in frag_generator:
        condition = True
        for function in filter_func_list:
            condition &= function(frag)
        if condition:
            yield frag


if __name__ == '__main__':
    slide_address = "../database_crawlers/bio_atlas_at_jake_gittlen_laboratories/data/1672.tiff"
    generator = generate_raw_fragments_from_zarr(zarr_loader(slide_address))
    for frag in filter_frag_from_generator(generator, [ThyroidFragmentFilters.empty_frag_with_laplacian]):
        print(frag.shape)

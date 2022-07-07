import pathlib

import matplotlib.pyplot as plt

from image_patcher import ImageAndSlidePatcher, ThyroidFragmentFilters

if __name__ == '__main__':
    res_patch_counts = []
    data_dir = "../national_cancer_institute/data/"
    for image_path in pathlib.Path(data_dir).glob("**/*.svs"):
        image_path = str(image_path)

        zarr_object = ImageAndSlidePatcher._zarr_loader(image_path)
        generator = ImageAndSlidePatcher._generate_raw_fragments_from_image_array_or_zarr(zarr_object)
        total_counts = ImageAndSlidePatcher._get_number_of_initial_frags(zarr_object=zarr_object)
        if generator is None:
            continue
        filters = [ThyroidFragmentFilters.empty_frag_with_laplacian_threshold]
        fragment_id = 0
        patch_filtered = ImageAndSlidePatcher._filter_frag_from_generator(generator, filters,
                                                                          all_frag_count=total_counts)
        for fragment, frag_pos in patch_filtered:
            fragment_id += 1
        res_patch_counts.append((fragment_id, total_counts))

    plt.hist([i[0] for i in res_patch_counts], bins=1000)
    plt.savefig("patch_distribution.jpeg")
    plt.clf()

    plt.hist([i[0] / i[1] for i in res_patch_counts], bins=1000)
    plt.savefig("patch_percent_distribution.jpeg")
    plt.clf()


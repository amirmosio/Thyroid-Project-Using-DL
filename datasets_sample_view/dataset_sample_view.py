# import libtiff
# import pytiff
import cv2
import tifffile


def show_tif_image(address, name, key=0, w_from=0, h_from=0, size=700, whole_image=False):
    import zarr
    image_zarr = tifffile.imread(address, aszarr=True, key=key)
    zarr = zarr.open(image_zarr, mode='r')
    if not whole_image:
        image_frag = zarr[w_from:min(w_from + size, zarr.shape[0]), h_from:min(h_from + size, zarr.shape[1])]
    else:
        image_frag = zarr[0:zarr.shape[0], 0:zarr.shape[1]]
    cv2.imshow(f"name:{name} - shape:{image_frag.shape} - page:{key}", image_frag)
    print(f"name: {name}, shape: {zarr.shape}")
    image_zarr.close()


def show_CAMELYON16_sample_view():
    # show_tif_image('data/CAMELYON16/tumor_084.tif', "CAMELYON16", key=7)
    show_tif_image('data/CAMELYON16/tumor_084.tif', "CAMELYON16", key=0, w_from=10000, h_from=50000)


def show_CAMELYON17_sample_view():
    show_tif_image('data/CAMELYON17/patient_083_node_4.tif', "CAMELYON17", key=7)


def show_Papsociety_sample_view():
    image_frag = cv2.imread(
        'data/Papsociety/Follicular_neoplasm2,_low_power,_confirmed_FVPTC_DQ_SM.jpg')
    cv2.imshow(f"Papsociety - {image_frag.shape}", image_frag)


def show_test():
    # show_tif_image('data/CAMELYON16/tumor_084.tif', "CAMELYON16", key=7)
    show_tif_image('data/test/1.tiff', "test", key=0, w_from=10000, h_from=30000, size=1000)


if __name__ == '__main__':
    show_CAMELYON16_sample_view()
    # show_CAMELYON17_sample_view()
    # show_Papsociety_sample_view()
    show_test()
    while True:
        if cv2.waitKey(1) == ord('q'):
            break

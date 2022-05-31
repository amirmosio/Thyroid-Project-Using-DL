# import cv2
# import tifffile
#
#
# def show_tif_image(address, name, key=0, w_from=0, h_from=0, size=700):
#     import zarr
#     image_zarr = tifffile.imread(address, aszarr=True, key=key)
#     zarr = zarr.open(image_zarr, mode='r')
#     image_frag = zarr[w_from:w_from + size, h_from:h_from + size]
#     cv2.imshow(f"name:{name} - shape:{image_frag.shape} - page:{key}", image_frag)
#     zarr.()
#     image_zarr.close()

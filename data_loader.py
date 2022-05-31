import tifffile
import zarr as ZarrObject


def zarr_loader(tiff_address, key=0):
    image_zarr = tifffile.imread(tiff_address, aszarr=True, key=key)
    zarr = ZarrObject.open(image_zarr, mode='r')
    return zarr

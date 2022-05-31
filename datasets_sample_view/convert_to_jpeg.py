from PIL import Image
import zarr
import tifffile

def convert_tif_to_jpeg():
    input_address = "data/test/1672.tiff"
    # outfile = "data/test/out.jpeg"
    outfile = "data/test/out.zarr"
    image_zarr = tifffile.imread(input_address, aszarr=True, key=0)
    zarr_image = zarr.open(image_zarr, mode='r')
    zarr.save(outfile, zarr_image)
    ## RAM PROBLEM
    # im = Image.open()
    # out = im.convert("RGB")
    # out.save(outfile, "JPEG", quality=90)


if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = 1000 * 1000 * 256 * 256
    convert_tif_to_jpeg()

# from dalib.translation.fourier_transform import FourierTransform
import importlib.util
import sys

from PIL import Image

fourier_transform_address = "E:\\Documentwork\\sharif\\CE Project\\future\\Thyroid Project\\Thyroid-Project-Using-DL\\classification_stuff\\Transfer-Learning-Library\\dalib\\translation\\fourier_transform.py"
spec = importlib.util.spec_from_file_location("module.name", fourier_transform_address)
foo = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = foo
spec.loader.exec_module(foo)
FourierTransform = foo.FourierTransform
image_list = ["bio_tile (1).jpeg", "bio_tile (2).jpeg", "bio_tile (3).jpeg", "bio_tile (4).jpeg", "bio_tile (4).jpeg"]
amplitude_dir = "amplitude_dir"
fourier_transform = FourierTransform(image_list, amplitude_dir, beta=0, rebuild=False)
source_image = Image.open("tile.jpeg")  # image form source domain
source_image_in_target_style = fourier_transform(source_image)

source_image_in_target_style.save("out_fda.jpeg")

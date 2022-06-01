import torchvision
from torch import nn

from classification_stuff.data_loader import DataLoaderUtil
from classification_stuff.slide_preprocess import generate_raw_fragments_from_zarr, ThyroidFragmentFilters, \
    filter_frag_from_generator
from database_crawlers.web_stain_sample import ThyroidType


class ThyroidClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_net_101 = torchvision.models.resnet101(pretrained=True, progress=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        res_net_output = self.res_net_101(x)
        return self.classifier(res_net_output)


bio_atlas_data_dir = "../database_crawlers/bio_atlas_at_jake_gittlen_laboratories/data/"



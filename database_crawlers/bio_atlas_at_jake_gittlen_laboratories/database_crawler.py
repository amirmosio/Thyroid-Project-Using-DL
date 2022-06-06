import ssl
import time
from urllib.parse import urlparse
from urllib.request import urlopen

import certifi
from bs4 import BeautifulSoup

from database_crawlers.web_stain_sample import StainType, WebStainWSIOneDIndex

orig_sslsocket_init = ssl.SSLSocket.__init__
ssl.SSLSocket.__init__ = lambda *args, cert_reqs=ssl.CERT_NONE, **kwargs: orig_sslsocket_init(*args,
                                                                                              cert_reqs=ssl.CERT_NONE,
                                                                                              **kwargs)


class BioAtlasAtJakeGittlenLaboratoriesImage(WebStainWSIOneDIndex):

    def __init__(self, database_name, image_id, image_web_label, report, stain_type, is_wsi):
        super().__init__(database_name, image_id, image_web_label, report, stain_type, is_wsi)

    def _get_tile_url(self, zoom, partition=None, i=None, j=None):
        return f"https://bio-atlas.psu.edu/human/tile.jpeg.php?s={self.image_id}&z={zoom}&i={partition}"

    def get_slide_view_url(self):
        return f"https://bio-atlas.psu.edu/human/view.php?s={self.image_id}"

    def _get_file_path_name(self):
        return self.save_path + self.image_id

    def find_best_zoom(self):
        return 0


class BioAtlasThyroidSlideProvider:
    page_link = "https://bio-atlas.psu.edu/human/search.php?q=Thyroid&organism%5B%5D=5&age_fr=&age_fr_units=1&age_to=&age_to_units=1&sex%5B%5D=all&thumbnails=on&rpp=30&as_sfid=AAAAAAW0RrspdnblpiFwz8osoAdvS8nafd1J9LG_ARQ-IF_NZ3aI2EXCMDBeqE_iD5rUo1QLg454tS63DMSgATSzgrksb4rMi-GWPl3O9f3JKlqGn8oXoqbOYok3__yZx69ewzg%3D&as_fid=6900aeb3e4cc9f39ef9738a2f11c2cefb8c3f37c#results"
    database_name = "BioAtlasThyroidSlideProvider"
    stain_type = StainType.H_AND_E
    is_wsi = True

    @classmethod
    def get_web_stain_samples(cls):
        print(cls.page_link)
        try:
            html_text = urlopen(cls.page_link).read()
            soup = BeautifulSoup(html_text, 'html.parser')
            search_results = soup.find_all("div", {"class": "shadow-box search-result-item search-result-slide"})
            for result_item in search_results:
                image_view_url = result_item.find("a").attrs['href']
                query_param = urlparse(image_view_url).query.split("=")
                if query_param[0] != "s": raise Exception("Query params does not contains image url")
                image_id = query_param[1]
                image_web_label = str(result_item.find("b", text="Diagnosis").next_sibling)
                yield BioAtlasAtJakeGittlenLaboratoriesImage(cls.database_name, image_id, image_web_label, None,
                                                             cls.stain_type, cls.is_wsi)
        except Exception as e:
            print(e)
            time.sleep(2)
            yield cls.get_web_stain_samples()


if __name__ == '__main__':
    bio_atlas_provider = BioAtlasThyroidSlideProvider()
    for slide in bio_atlas_provider.get_web_stain_samples():
        print(slide.image_id, slide.image_web_label, slide.get_slide_view_url())
        slide.crawl_image_save_jpeg_and_json()

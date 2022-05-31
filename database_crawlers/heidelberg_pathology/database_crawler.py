import time
from urllib.parse import urlparse
from urllib.request import urlopen

from bs4 import BeautifulSoup

from database_crawlers.web_stain_sample import StainType, WebStainWSITwoDIndex


class HeidelbergPathologyImage(WebStainWSITwoDIndex):

    def __init__(self, database_name, image_id, image_web_label, report, stain_type, is_wsi):
        super().__init__(database_name, image_id, image_web_label, report, stain_type, is_wsi)

    def _get_tile_url(self, zoom, partition=None, i=None, j=None):
        return f"https://eliph.klinikum.uni-heidelberg.de/dzi/atlas/05-schilddruese/05-{'%.2d' % int(self.image_id)}_files/{zoom}/{i}_{j}.jpeg"

    def get_slide_view_url(self):
        return f"https://eliph.klinikum.uni-heidelberg.de/atlas/?c=05-schilddruese&context=image&pg={self.image_id}"

    def _get_file_path_name(self):
        return self.save_path + self.image_id

    def find_best_zoom(self):
        # 16 -> 0
        return 16


class HeidelbergPathologyProvider:
    page_link = "https://eliph.klinikum.uni-heidelberg.de/atlas/?c=05-schilddruese&context=image"
    database_name = "HeidelbergPathology"
    stain_type = StainType.H_AND_E
    is_wsi = True

    @classmethod
    def get_web_stain_samples(cls):
        print(cls.page_link)
        try:
            html_text = urlopen(cls.page_link).read()
            soup = BeautifulSoup(html_text, 'html.parser')
            search_results = soup.find_all("div", {"class": "casegrid"})
            for result_item in search_results:
                image_view_url = result_item.find("a").attrs['href']
                query_param = urlparse(image_view_url).query.split("=")
                if "image&pg" not in query_param: raise Exception("Query params does not contains image id")
                image_id = query_param[-1]
                image_web_label = str(result_item.find("b").next)
                yield HeidelbergPathologyImage(cls.database_name, image_id, image_web_label, None,
                                               cls.stain_type, cls.is_wsi)
        except Exception as e:
            print(e)
            time.sleep(2)
            yield cls.get_web_stain_samples()


if __name__ == '__main__':
    bio_atlas_provider = HeidelbergPathologyProvider()
    for slide in bio_atlas_provider.get_web_stain_samples():
        print(slide.image_id, slide.image_web_label, slide.get_slide_view_url())
        slide.crawl_image_save_jpeg_and_json()
        break

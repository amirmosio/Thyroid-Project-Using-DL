import json
from urllib.parse import urlparse
from urllib.request import urlretrieve

import requests
from bs4 import BeautifulSoup

from database_crawlers.web_stain_sample import WebStainSample, StainType


class StanfordTissueMicroArrayStainSample(WebStainSample):

    def __init__(self, database_name, image_id, image_web_label, report, stain_type, is_wsi):
        super().__init__(database_name, image_id, image_web_label, report, stain_type, is_wsi)

    def get_slide_view_url(self):
        return f"https://storage.googleapis.com/jpg.tma.im/{self.image_id}"

    def get_file_name(self):
        image_raw_id = self.image_id.replace("/", "_")
        image_raw_id = ".".join(image_raw_id.split(".")[:len(image_raw_id.split(".")) - 1])
        return self.save_path + image_raw_id

    def get_relative_image_path(self):
        return self.get_file_name() + ".jpeg"

    def get_relative_json_path(self):
        return self.get_file_name() + ".json"

    def crawl_image_save_jpeg(self):
        urlretrieve(self.get_slide_view_url(), self.get_relative_image_path())
        json_object = json.dumps(self.to_json())
        with open(self.get_relative_json_path(), "w") as outfile:
            outfile.write(json_object)


class StanfordTissueMicroArraySlideProvider:
    page_link = "https://tma.im/cgi-bin/selectImages.pl?organ=thyroid"
    database_name = "StanfordTissueMicroArray"
    stain_type = StainType.UNKNOWN
    is_wsi = False

    @classmethod
    def get_web_stain_samples(cls):
        payload = {'250 small images': '250 small images'}
        files = []
        headers = {
            'Cookie': 'DAD_ATTEMPTS=0; DAD_SID=36d77eb69e009b1cf1ebc9c3d7866546; DAD_USERID=WORLD'
        }
        html_text = requests.post(cls.page_link, files=files, headers=headers, data=payload).content.decode("utf-8")
        soup = BeautifulSoup(html_text, 'html.parser')
        search_results = soup.find_all("div", {"class": "iDiv0", "style": "width: 86px; height: 260px;"})
        for result_item in search_results:
            image_url = result_item.find("a", {"target": "_blank"}).attrs['href']
            image_id = "/".join(urlparse(image_url).path.strip("/").split("/")[1:])
            image_web_label = list(result_item.find_all("p", {"class": "iDiv1"}))[-2].text
            yield StanfordTissueMicroArrayStainSample(cls.database_name, image_id, image_web_label, None,
                                                      cls.stain_type, cls.is_wsi)


if __name__ == '__main__':
    for slide in StanfordTissueMicroArraySlideProvider.get_web_stain_samples():
        print(slide.image_id, slide.image_web_label, slide.get_slide_view_url())
        slide.crawl_image_save_jpeg()

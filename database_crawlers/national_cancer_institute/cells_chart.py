import pathlib

import matplotlib.pyplot as plt

from national_cancer_institute.read_xml_file import get_slide_info_from_bcr_xml

if __name__ == '__main__':
    data_dir = "data/"
    slide_infos = {}
    for xml_path in pathlib.Path(data_dir).glob("**/*.xml"):
        slide_infos.update(get_slide_info_from_bcr_xml(str(xml_path)))
    print(slide_infos)
    normal_percents = [int(item[1]) for item in slide_infos.values()]
    plt.hist(normal_percents, bins=100)
    plt.savefig("cell_distribution.jpeg")

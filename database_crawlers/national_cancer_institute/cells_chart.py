import pathlib

import matplotlib.pyplot as plt

from national_cancer_institute.read_xml_file import get_slide_info_from_bcr_xml

if __name__ == '__main__':
    data_dir = "data/"
    slide_infos = {}
    for xml_path in pathlib.Path(data_dir).glob("**/*.xml"):
        slide_infos.update(get_slide_info_from_bcr_xml(str(xml_path)))
    cell_percents = [int(item[1]) for item in slide_infos.values() if int(item[2]) == 0]
    print("tumor:", len([i for i in cell_percents if i == 100]))
    print("normal", len([i for i in cell_percents if i == 0]))
    print(len(cell_percents))
    plt.hist(cell_percents, bins=150)
    plt.savefig("tumor_cell_distribution.jpeg")

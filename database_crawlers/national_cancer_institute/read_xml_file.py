from xml.dom import minidom


def get_slide_info_from_bcr_xml(xml_path):
    file = minidom.parse(xml_path)
    patient = file.childNodes[0].getElementsByTagName("bio:patient")[0]
    data_dict = {}
    try:
        for i in range(10):
            percent_tumor_cells = patient.getElementsByTagName("bio:percent_tumor_cells")[i].childNodes[
                0].data.strip()
            percent_normal_cells = patient.getElementsByTagName("bio:percent_normal_cells")[i].childNodes[
                0].data.strip()
            percent_stormal_cells = patient.getElementsByTagName("bio:percent_stromal_cells")[i].childNodes[
                0].data.strip()
            slide_barcode = patient.getElementsByTagName("shared:bcr_slide_barcode")[i].childNodes[0].data.strip()
            data_dict[slide_barcode] = (percent_normal_cells, percent_tumor_cells, percent_stormal_cells)
    except Exception as e:
        pass
    return data_dict


if __name__ == '__main__':
    path = "../national_cancer_institute/data/1aea8f2a-f809-4f19-bed3-1365e9aab33b/nationwidechildrens.org_biospecimen.TCGA-BJ-A28X.xml"
    res = get_slide_info_from_bcr_xml(path)
    print(res)

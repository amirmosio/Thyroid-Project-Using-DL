from xml.dom import minidom

if __name__ == '__main__':
    sample_file = "../national_cancer_institute/data/1aea8f2a-f809-4f19-bed3-1365e9aab33b/nationwidechildrens.org_biospecimen.TCGA-BJ-A28X.xml"
    file = minidom.parse(sample_file)
    models = file.getElementsByTagName('model')
    patient = file.childNodes[0].getElementsByTagName("bio:patient")[0]
    case_barcode = patient.getElementsByTagName("shared:bcr_patient_barcode")[0].childNodes[0].data.strip()
    for i in range(10):
        percent_tumor_cells = patient.getElementsByTagName("bio:percent_tumor_cells")[i].childNodes[
            0].data.strip()
        percent_normal_cells = patient.getElementsByTagName("bio:percent_normal_cells")[i].childNodes[
            0].data.strip()
        percent_stormal_cells = patient.getElementsByTagName("bio:percent_stromal_cells")[i].childNodes[
            0].data.strip()
        slide_barcode = patient.getElementsByTagName("shared:bcr_slide_barcode")[i].childNodes[0].data.strip()
        print(percent_normal_cells, percent_stormal_cells, percent_tumor_cells, slide_barcode)

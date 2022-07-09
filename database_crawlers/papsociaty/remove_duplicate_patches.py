import os
import shutil
if __name__ == '__main__':
    duplicate_info_file_path = "duplicate_image.txt"
    with open(duplicate_info_file_path, "r") as file:
        for line in file.readlines():
            folder_id = line.split(",")[0]
            folder_path = os.path.join("./patches", folder_id)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print("deleted")
            else:
                print("no")

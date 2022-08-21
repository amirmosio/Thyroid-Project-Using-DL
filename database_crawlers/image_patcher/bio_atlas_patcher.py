from image_patcher import ImageAndSlidePatcher

if __name__ == '__main__':
    database_folder_name = "bio_atlas_at_jake_gittlen_laboratories"
    database_directory = "../"
    image_slide_patcher = ImageAndSlidePatcher()
    image_slide_patcher.save_patches_in_folders(database_directory, database_folder_name)

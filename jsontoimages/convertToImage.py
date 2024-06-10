import os
import shutil
import time 

directory = "../"
json_path = "CalPolyImages2/"
folder_convert = "dataset_example"

sidewalk_dir = "../../CalPolyRoadDataset/sidewalk/"
sidewalk_mask_dir = "../../CalPolyRoadDataset/sidewalkMask/"

def execute_convert_command():
    ''' 
        Converts the CalPolyRoadImages2 into a Image + Mask, which will then be inserted 
        into the Dataset

        @params: None 
        @return: None
    ''' 

    os.chdir(directory)
    json_dir = os.listdir(json_path)
    os.chdir(json_path)

    for single_file in json_dir: 
        if single_file.endswith(".json"):
            file_name = os.path.splitext(single_file)[0]

            os.system(f"labelme_export_json {single_file} -o {folder_convert}")
            os.chdir(f"{folder_convert}")
            time.sleep(5)
            
            # img.png 
            # label.png
            os.rename('label.png', f'{file_name}.png')
            shutil.move(f'{file_name}.png', sidewalk_mask_dir)
            
            os.rename('img.png', f'{file_name}.png')
            shutil.move(f'{file_name}.png', sidewalk_dir)

            # Go Home
            os.chdir("../")



execute_convert_command()
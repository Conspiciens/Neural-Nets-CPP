import os

from PIL import Image 

class data_processing: 
    def __init__(self): 
        self.image_path = "/Volumes/Heavy_Duty/logging/image_data/" 
        self.logging_path = "/Volumes/Heavy_Duty/logging/logging_data/"
        self.merge_log_file = "/Volumes/Heavy_Duty/logging/"
        self.merge_log_files() 

    def merge_log_files(self) -> None: 
        file_count = 0 

        # Check if the logging path exists
        if not os.path.exists(self.logging_path):
            print("Logging path doesn't exist")
            raise IOError("%s: %s" % (self.logging_path, "Logging path doesn't exist")) 

        if not os.path.exists(self.merge_log_file):
            print("Directory for merged log file doesn't exist")
            raise IOError("%s: %s" % (self.merge_log_file, "Merge file directory doesn't exist"))
        
        if os.path.exists(self.merge_log_file + "merged_log_file.txt") == True: 
            print("Merge file exists already, returning") 
            return

        # Check whether path exists and file isn't a hidden file
        for files in os.listdir(self.logging_path): 
            file = os.path.join(self.logging_path, files) 
            if os.path.exists(file) and not files.startswith("."): 
                file_count += 1

        # Create merged log file
        merged_log_dir = os.path.join(self.merge_log_file, "merged_log_file.txt") 
        merged_log_file = open(merged_log_dir, 'w') 

        i = 0
        # Iterate through the log files 
        while i < file_count: 
            log_file_name = os.path.join(self.logging_path, "log_file_" + str(i) + ".txt") 
            log_file = open(log_file_name) 
        
            # Write data from the log file to merged file
            for idx, line in enumerate(log_file): 
                if idx == 0: 
                    continue 
                merged_log_file.write(line) 
                 
            log_file.close() 
            i += 1

        merged_log_file.close() 

            
            

    def __getitem__(self, idx) -> tuple[Image, Image, Image, int]:
        
        # Check if Image path and Logging path 
        if not os.path.exists(self.image_path): 
            print("Image path doesn't exist") 
            raise IOError("%s: %s" % (self.image_path, "Image path doesn't exist")) 

        if not os.path.exists(self.merge_log_file + "merged_log_file.txt"): 
            print("Logging path doesn't exist")  
            raise IOError("%s: %s" % (self.logging_path, "Logging path doesn't exist")) 
        
        line_req = None
        log_file = open(self.merge_log_file + "merged_log_file.txt"); 
        for line_idx, line in enumerate(log_file): 
            if line_idx - 1 == idx: 
                line_req = line            
                break 
        
        if line_req is None: 
            raise IndexError("No Index Found")

        # Split the line whenver there's a space
        line_arr = line_req.split(" ") 
        
        front_img_name = line_arr[1].split("/")[-1] 
        left_img_name = line_arr[2].split("/")[-1] 
        right_img_name = line_arr[3].split("/")[-1] 
        
        front_img_name = self.image_path + front_img_name 
        left_img_name = self.image_path + left_img_name
        right_img_name = self.image_path + right_img_name

        # Get all Images and steering value 
        front_image = Image.open(front_img_name)
        right_image = Image.open(right_img_name)  
        left_image = Image.open(left_img_name)
    
        front_image_yuv = front_image.convert('YCbCr') 
        right_image_yuv = right_image.convert('YCbCr') 
        left_image_yuv = left_image.convert('YCbCr')
        steering_val = line_arr[4] 

        return front_image_yuv, right_image_yuv, left_image_yuv, steering_val 
        

if __name__ == '__main__':
    preprocess = data_processing()

import os
import shutil

current_folder = os.getcwd()

folder_name = "instances"
folder_path = os.path.join(current_folder, folder_name)

if os.path.exists(folder_path) and os.path.isdir(folder_path):
    shutil.rmtree(folder_path)
    print(f"instances directory removed")
else:
    print("no instances directory")
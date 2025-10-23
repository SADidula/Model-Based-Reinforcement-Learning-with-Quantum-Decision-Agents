import json
import os
from ament_index_python.packages import get_package_share_directory

class RoboticConfig:
    def __init__(self, file_name: str, path="config/", package_name="ursina_sim"):                
        # Resolve absolute path inside the installed share dir
        pkg_share = get_package_share_directory(package_name)
        self.abs_path = os.path.join(pkg_share, path + file_name)
        if not os.path.exists(self.abs_path):
            raise FileNotFoundError(f"Config not found: {self.abs_path}")
             
    def load_config(self):
        with open(self.abs_path, "r") as f:
            return json.load(f)   
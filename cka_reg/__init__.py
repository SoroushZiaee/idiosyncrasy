import cka_reg
from pathlib import Path
import os, platform, getpass


PROJECT_ROOT = str(Path(cka_reg.__file__).parents[1])
# print(f"{PROJECT_ROOT = }")
IMAGENET_PATH = "/home/soroush1/projects/def-kohitij/soroush1/idiosyncrasy/imagenet"

# print(f"{platform.node() = }")

if platform.node().startswith("node"):
    # DATA_PATH = f"{PROJECT_ROOT}/data"
    DATA_PATH = f"/home/soroush1/scratch/idiosyncrasy/"
else:
    DATA_PATH = f"/home/soroush1/scratch/idiosyncrasy/"
    # print(f"{DATA_PATH = }")


from .utils import *

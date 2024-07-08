import cka_reg
from pathlib import Path
import os, platform, getpass


PROJECT_ROOT = str(Path(cka_reg.__file__).parents[1])
IMAGENET_PATH = "/datashare/ImageNet/ILSVRC2012"


if platform.node().startswith("node"):
    DATA_PATH = f"{PROJECT_ROOT}/data"
else:
    DATA_PATH = f"/braintree/home/{getpass.getuser()}/data/from_dapello"


# from .utils import *

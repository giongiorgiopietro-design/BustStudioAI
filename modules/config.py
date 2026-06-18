from dataclasses import dataclass

@dataclass
class Config:

    DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

    OUTPUT_FOLDER = "outputs"

    NORMAL_STRENGTH = 2.0

    HYPERDETAIL = 1.8

    SAVE_16BIT = True

cfg = Config()

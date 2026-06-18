from transformers import pipeline
from PIL import Image
import numpy as np


class DepthEngine:

    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        self.model_name = model_name
        self.pipe = None

    def load(self):
        print("Caricamento modello...")
        self.pipe = pipeline(
            "depth-estimation",
            model=self.model_name
        )
        print("✅ Modello caricato")

    def generate(self, image):

        if self.pipe is None:
            raise RuntimeError("Il modello non è stato caricato. Eseguire load().")

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        depth = self.pipe(image)["depth"]

        return depth

    def save(self, depth, output_path="outputs/depth.png"):

        depth.save(output_path)

        print(f"✅ Salvata: {output_path}")

    def generate_depth16(self, depth):

        depth_np = np.array(depth).astype(np.float32)

        depth_np -= depth_np.min()

        depth_np /= depth_np.max()

        depth16 = (depth_np * 65535).astype(np.uint16)

        return depth16

    def save_depth16(self, depth16, output_path="outputs/depth16.png"):

        Image.fromarray(depth16).save(output_path)

        print(f"✅ Salvata: {output_path}")

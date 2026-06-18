from dataclasses import dataclass, field
from PIL import Image
import numpy as np


@dataclass
class Project:

    image_path: str

    original: Image.Image = None
    original_np: np.ndarray = None

    depth = None
    depth16 = None
    normal = None
    hyperdepth = None

    restored = None
    background = None

    report: dict = field(default_factory=dict)

    def load_image(self):

        self.original = Image.open(self.image_path).convert("RGB")
        self.original_np = np.array(self.original)

        self.report["image"] = self.image_path
        self.report["width"] = self.original.width
        self.report["height"] = self.original.height

        print(f"✅ Immagine caricata ({self.original.width}x{self.original.height})")

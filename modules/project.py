from dataclasses import dataclass
from PIL import Image
import numpy as np


@dataclass
class Project:

    # Percorso immagine
    image_path: str = ""

    # Immagine originale
    original: Image.Image = None

    # Versione numpy
    original_np: np.ndarray = None

    # Output pipeline
    restored = None
    background = None

    depth = None
    depth16 = None

    normal = None

    hyperdepth = None

    displacement = None

    report = {}

    def load_image(self):

        self.original = Image.open(self.image_path).convert("RGB")

        self.original_np = np.array(self.original)

        print("✅ Immagine caricata")

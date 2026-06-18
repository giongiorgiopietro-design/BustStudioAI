from modules.config import cfg
from modules.utils import ensure_folder

from PIL import Image
import numpy as np
import cv2


class HyperDepthEngine:

    def __init__(self):
        pass

    def generate(self, depth16):

        # Conversione in float
        depth = depth16.astype(np.float32) / 65535.0

        # Base morbida
        blur = cv2.GaussianBlur(depth, (0, 0), 3)

        # Estrazione dettagli
        detail = depth - blur

        # Potenziamento dettagli
        hyper = depth + detail * cfg.HYPERDETAIL

        # Limiti
        hyper = np.clip(hyper, 0.0, 1.0)

        return hyper

    def save(self, hyperdepth, output_path=None):

        ensure_folder(cfg.OUTPUT_FOLDER)

        if output_path is None:
            output_path = f"{cfg.OUTPUT_FOLDER}/hyperdepth.png"

        hyper16 = (hyperdepth * 65535).astype(np.uint16)

        Image.fromarray(hyper16).save(output_path)

        print(f"✅ Salvata: {output_path}")

    def run(self, project):

        project.hyperdepth = self.generate(project.depth16)

        self.save(project.hyperdepth)

        return project

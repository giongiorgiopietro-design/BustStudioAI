from modules.config import cfg
from modules.utils import ensure_folder

from PIL import Image
import numpy as np
import cv2


class NormalEngine:

    def __init__(self):
        pass

    def generate(self, depth16):

        depth = depth16.astype(np.float32) / 65535.0

        strength = cfg.NORMAL_STRENGTH

        dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)

        dx *= strength
        dy *= strength

        normal = np.dstack((-dx, -dy, np.ones_like(depth)))

        normal /= np.linalg.norm(normal, axis=2, keepdims=True)

        normal = ((normal + 1.0) / 2.0 * 255).astype(np.uint8)

        return normal

    def save(self, normal, output_path=None):

        ensure_folder(cfg.OUTPUT_FOLDER)

        if output_path is None:
            output_path = f"{cfg.OUTPUT_FOLDER}/normal.png"

        Image.fromarray(normal).save(output_path)

        print(f"✅ Salvata: {output_path}")

    def run(self, project):

        project.normal = self.generate(project.depth16)

        self.save(project.normal)

        return project

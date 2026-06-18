from modules.project import Project
from modules.depth import DepthEngine
from modules.normal import NormalEngine
from modules.hyperdepth import HyperDepthEngine


def main():

    project = Project(image_path="foto.jpg")

    project.load_image()

    depth = DepthEngine()
    depth.run(project)

    normal = NormalEngine()
    normal.run(project)

    hyper = HyperDepthEngine()
    hyper.run(project)

    print("\n==============================")
    print(" BustStudioAI")
    print("==============================")
    print("✅ Pipeline completata")
    print("📁 File salvati in outputs/")


if __name__ == "__main__":
    main()

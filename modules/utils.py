import os

def ensure_folder(folder):

    os.makedirs(folder, exist_ok=True)

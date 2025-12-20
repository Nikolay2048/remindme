from pathlib import Path
from time import time, sleep


def wait_for_file(path: Path, timeout=10):
    """Ожидание, пока файл появится на диске"""
    start = time()
    while not path.exists():
        sleep(0.2)
        if time() - start > timeout:
            raise FileNotFoundError(f"file {path} not found after {timeout} seconds")

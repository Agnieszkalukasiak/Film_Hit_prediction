import subprocess

def ensure_lfs_files():
    subprocess.run(["git", "lfs", "pull"])
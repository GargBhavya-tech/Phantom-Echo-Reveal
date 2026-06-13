import os

def create_structure(base_path: str):
    dirs = ["color", "depth", "pose"]
    for d in dirs:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)
    
    print(f"Dataset structure created at: {base_path}")
    print("Please place your files here:")
    print("  color/ -> rgb images (e.g., 00000.jpg)")
    print("  depth/ -> depth maps (e.g., 00000.npy or 16-bit .png)")
    print("  pose/  -> camera poses (e.g., 00000.txt as 4x4 matrix)")

if __name__ == "__main__":
    create_structure("data/my_real_scene")

import os
from PIL import Image

def create_animation(directory='figures', output_filename='figures/oras5_djf_animation.gif', duration=500):
    """
    Creates an animated GIF from ORAS5 DJF 6-panel PNG files.
    """
    # Find all files matching the pattern: oras5_djf_YYYY_6panels.png
    files = [f for f in os.listdir(directory) if f.startswith('oras5_djf_avg_') and f.endswith('_6panels.png')]
    
    # Sort files by year (the 3rd element when splitting by '_')
    files.sort(key=lambda x: int(x.split('_')[3]))
    
    if not files:
        print(f"Error: No matching files found in {directory}")
        return

    print(f"Found {len(files)} frames. Processing...")

    images = []
    for filename in files:
        path = os.path.join(directory, filename)
        # Open and convert to RGB (standard for GIF)
        img = Image.open(path).convert('RGB')
        images.append(img)

    # Save as animated GIF
    images[0].save(
        output_filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"Success! Animation saved to {output_filename}")

if __name__ == "__main__":
    create_animation()

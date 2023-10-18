"""Merge frames (manually extracted from simulated_annealing.gif) into a GIF file."""

from pathlib import Path

from PIL import Image


if __name__ == '__main__':
    FILE_DIR = Path(__file__).parent

    print(f'Creating GIF animation out of .tiff frames in {FILE_DIR}')
    folder = Path(FILE_DIR / 'avoiding local optima frames')
    frames = [Image.open(img) for img in sorted(folder.glob('*.tiff'))]

    save_to = FILE_DIR.parent / 'avoiding_local_optima.gif'
    
    frames[0].save(
        save_to,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=1_000,
        loop=0,
    )

    print(f'Saved to {save_to}')

import subprocess
import os

def webm_to_gif(input_path, output_path=None, fps=5, scale=480):
    """
    Convert a .webm video to an optimized GIF using FFmpeg palette technique.
    
    Args:
        input_path (str): Full path to input .webm file
        output_path (str): Full path to save gif (optional, same name as input if None)
        fps (int): Frames per second for gif
        scale (int): Output width (height auto-calculated)
    """
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".gif"

    palette = "palette.png"

    # Step 1: generate palette
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"fps={fps},scale={scale}:-1:flags=lanczos,palettegen",
        palette
    ], check=True)

    # Step 2: create gif
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path, "-i", palette,
        "-filter_complex", f"fps={fps},scale={scale}:-1:flags=lanczos[x];[x][1:v]paletteuse",
        output_path
    ], check=True)

    os.remove(palette)
    print(f"✅ GIF saved: {output_path}")


# Example usage with full path
webm_to_gif("automama_run.mp4", "automama_run.gif", fps=10, scale=123)

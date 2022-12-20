import typer
from pathlib import Path

from generate import generate

app = typer.Typer()


@app.command()
def generate_compositions(
    backgrounds_dir: Path = typer.Option(
        ...,
        dir_okay=True,
        resolve_path=True,
        help="Path to the directory with downloaded screenshots which will be used as background.",
    ),
    compositions_dir: Path = typer.Option(
        ...,
        dir_okay=True,
        resolve_path=True,
        help="Path to the directory with compositions of backgrounds and logos.",
    ),
    objects_dir: Path = typer.Option(
        ...,
        dir_okay=True,
        resolve_path=True,
        help="Path to the directory with objects for which detection model will be trained.",
    ),
    masks_dir: Path = typer.Option(
        ...,
        dir_okay=True,
        resolve_path=True,
        help="Path to the directory with binary masks for logos.",
    ),
    yolo_labels_dir: Path = typer.Option(
        ...,
        dir_okay=True,
        resolve_path=True,
        help="Path to the directory with labels for training yolo model.",
    ),
    n_samples: int = typer.Option(
        500,
        help="Number of training samples to produce.",
    ),

):

    generate(
        backgrounds_dir,
        objects_dir,
        masks_dir,
        compositions_dir,
        yolo_labels_dir,
        n_samples
    )


if __name__ == "__main__":
    app()

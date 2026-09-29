from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imagehash

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "data" / "gvlid_hybrid_processed" / "train"
TEST_DIR = ROOT / "data" / "gvlid_natural_eval" / "test"

OUTPUT_DIR = ROOT / "near_duplicates_review"

NATURAL_PREFIX = "gvlid_"

# Distancia máxima considerada sospechosa
THRESHOLD = 5

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PREVIEW_SIZE = (500, 500)


def get_images(directory, natural_only=False):
    images = []

    for path in directory.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        if natural_only and not path.name.startswith(NATURAL_PREFIX):
            continue

        images.append(path)

    return sorted(images)


def calculate_phash(path):
    try:
        with Image.open(path) as image:
            return imagehash.phash(image.convert("RGB"))
    except Exception as e:
        print(f"Error procesando {path}: {e}")
        return None


def get_class_name(path):
    return path.parent.name


def prepare_preview(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(PREVIEW_SIZE)

        canvas = Image.new("RGB", PREVIEW_SIZE, "white")

        x = (PREVIEW_SIZE[0] - image.width) // 2
        y = (PREVIEW_SIZE[1] - image.height) // 2

        canvas.paste(image, (x, y))

        return canvas


def create_comparison(index, test_path, train_path, distance):
    test_image = prepare_preview(test_path)
    train_image = prepare_preview(train_path)

    width = PREVIEW_SIZE[0] * 2
    header_height = 130
    height = PREVIEW_SIZE[1] + header_height

    comparison = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    comparison.paste(
        test_image,
        (0, header_height)
    )

    comparison.paste(
        train_image,
        (PREVIEW_SIZE[0], header_height)
    )

    draw = ImageDraw.Draw(comparison)

    test_class = get_class_name(test_path)
    train_class = get_class_name(train_path)

    draw.text(
        (20, 10),
        f"PAR SOSPECHOSO #{index}",
        fill="black"
    )

    draw.text(
        (20, 40),
        f"Distancia pHash: {distance}",
        fill="black"
    )

    draw.text(
        (20, 75),
        f"TEST: {test_class}",
        fill="black"
    )

    draw.text(
        (PREVIEW_SIZE[0] + 20, 75),
        f"TRAIN: {train_class}",
        fill="black"
    )

    draw.text(
        (20, 100),
        test_path.name,
        fill="black"
    )

    draw.text(
        (PREVIEW_SIZE[0] + 20, 100),
        train_path.name,
        fill="black"
    )

    output_path = (
        OUTPUT_DIR /
        f"pair_{index:03d}_distance_{distance}.jpg"
    )

    comparison.save(
        output_path,
        quality=95
    )


def main():
    print("Buscando imágenes...")

    train_images = get_images(
        TRAIN_DIR,
        natural_only=True
    )

    test_images = get_images(TEST_DIR)

    print(
        f"Imágenes naturales de entrenamiento: "
        f"{len(train_images)}"
    )

    print(
        f"Imágenes naturales de test: "
        f"{len(test_images)}"
    )

    print("\nCalculando perceptual hashes...")

    train_hashes = []

    for path in train_images:
        hash_value = calculate_phash(path)

        if hash_value is not None:
            train_hashes.append(
                (path, hash_value)
            )

    test_hashes = []

    for path in test_images:
        hash_value = calculate_phash(path)

        if hash_value is not None:
            test_hashes.append(
                (path, hash_value)
            )

    print("\nComparando train vs test...")

    suspicious = []

    for test_path, test_hash in test_hashes:

        best_distance = None
        best_train_path = None

        for train_path, train_hash in train_hashes:

            distance = test_hash - train_hash

            if (
                best_distance is None
                or distance < best_distance
            ):
                best_distance = distance
                best_train_path = train_path

        if (
            best_distance is not None
            and best_distance <= THRESHOLD
        ):
            suspicious.append(
                (
                    test_path,
                    best_train_path,
                    best_distance
                )
            )

    print("\n==============================")
    print("RESULTADOS")
    print("==============================")

    print(f"Train natural: {len(train_hashes)}")
    print(f"Test natural: {len(test_hashes)}")
    print(f"Threshold pHash: {THRESHOLD}")
    print(
        f"Pares sospechosos: "
        f"{len(suspicious)}"
    )

    if not suspicious:
        print(
            "\nNo se encontraron near-duplicates."
        )
        return

    # Crear carpeta de revisión
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nGenerando comparaciones visuales..."
    )

    for index, (
        test_path,
        train_path,
        distance
    ) in enumerate(suspicious, start=1):

        create_comparison(
            index,
            test_path,
            train_path,
            distance
        )

        print(
            f"Par {index}: "
            f"distancia {distance}"
        )

    print("\n==============================")
    print("REVISIÓN VISUAL GENERADA")
    print("==============================")

    print(
        f"\nCarpeta:\n{OUTPUT_DIR}"
    )

    print(
        f"\nSe generaron "
        f"{len(suspicious)} comparaciones."
    )


if __name__ == "__main__":
    main()
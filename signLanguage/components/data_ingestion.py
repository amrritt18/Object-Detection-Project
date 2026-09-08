from pathlib import Path
import random
import shutil


class DataIngestion:
    def __init__(
        self,
        dataset_path: str,
        train_ratio: float = 0.80,
        valid_ratio: float = 0.10,
        test_ratio: float = 0.10,
        random_state: int = 42,
    ):
        self.dataset_path = Path(dataset_path)

        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        self.train_images = self.dataset_path / "train" / "images"
        self.train_labels = self.dataset_path / "train" / "labels"

        self.valid_images = self.dataset_path / "valid" / "images"
        self.valid_labels = self.dataset_path / "valid" / "labels"

        self.test_images = self.dataset_path / "test" / "images"
        self.test_labels = self.dataset_path / "test" / "labels"

    def _validate_ratios(self) -> None:
        total_ratio = (
            self.train_ratio
            + self.valid_ratio
            + self.test_ratio
        )

        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                "train_ratio, valid_ratio and test_ratio "
                "must sum to 1.0"
            )

    def _create_directories(self) -> None:
        directories = [
            self.valid_images,
            self.valid_labels,
            self.test_images,
            self.test_labels,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _get_image_files(self) -> list[Path]:
        supported_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        image_files = [
            file
            for file in self.train_images.iterdir()
            if file.is_file()
            and file.suffix.lower() in supported_extensions
        ]

        return image_files

    def _validate_image_label_pairs(
        self,
        image_files: list[Path],
    ) -> list[Path]:
        valid_files = []

        for image_file in image_files:
            label_file = self.train_labels / f"{image_file.stem}.txt"

            if label_file.exists():
                valid_files.append(image_file)
            else:
                print(
                    f"Warning: Label not found for "
                    f"{image_file.name}"
                )

        return valid_files

    def _move_files(
        self,
        image_files: list[Path],
        destination_images: Path,
        destination_labels: Path,
    ) -> None:

        for image_file in image_files:
            label_file = self.train_labels / f"{image_file.stem}.txt"

            shutil.move(
                str(image_file),
                str(destination_images / image_file.name),
            )

            shutil.move(
                str(label_file),
                str(destination_labels / label_file.name),
            )

    def split_dataset(self) -> None:
        self._validate_ratios()

        if not self.train_images.exists():
            raise FileNotFoundError(
                f"Training images directory not found: "
                f"{self.train_images}"
            )

        if not self.train_labels.exists():
            raise FileNotFoundError(
                f"Training labels directory not found: "
                f"{self.train_labels}"
            )

        self._create_directories()

        image_files = self._get_image_files()

        if not image_files:
            raise ValueError(
                "No images found in the training directory."
            )

        image_files = self._validate_image_label_pairs(
            image_files
        )

        if not image_files:
            raise ValueError(
                "No valid image-label pairs found."
            )

        random.seed(self.random_state)
        random.shuffle(image_files)

        total_images = len(image_files)

        train_end = int(
            total_images * self.train_ratio
        )

        valid_end = train_end + int(
            total_images * self.valid_ratio
        )

        valid_files = image_files[train_end:valid_end]
        test_files = image_files[valid_end:]

        self._move_files(
            valid_files,
            self.valid_images,
            self.valid_labels,
        )

        self._move_files(
            test_files,
            self.test_images,
            self.test_labels,
        )

        remaining_train_images = len(
            list(self.train_images.iterdir())
        )

        print("\nDataset splitting completed.")
        print(f"Total images : {total_images}")
        print(f"Train images : {remaining_train_images}")
        print(f"Valid images : {len(valid_files)}")
        print(f"Test images  : {len(test_files)}")


if __name__ == "__main__":

    dataset_path = (
        r"D:\Object-Detection-Project"
        r"\data\Sign detection.v5i.yolov11"
    )

    data_ingestion = DataIngestion(
        dataset_path=dataset_path,
        train_ratio=0.80,
        valid_ratio=0.10,
        test_ratio=0.10,
        random_state=42,
    )

    data_ingestion.split_dataset()
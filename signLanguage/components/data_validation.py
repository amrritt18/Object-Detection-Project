from pathlib import Path

import cv2
import yaml


class DataValidation:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

        self.train_images = self.dataset_path / "train" / "images"
        self.train_labels = self.dataset_path / "train" / "labels"

        self.valid_images = self.dataset_path / "valid" / "images"
        self.valid_labels = self.dataset_path / "valid" / "labels"

        self.test_images = self.dataset_path / "test" / "images"
        self.test_labels = self.dataset_path / "test" / "labels"

        self.data_yaml = self.dataset_path / "data.yaml"

    def _check_directory(self, directory: Path) -> bool:
        if not directory.exists():
            print(f"Directory not found: {directory}")
            return False

        if not directory.is_dir():
            print(f"Path is not a directory: {directory}")
            return False

        return True

    def _check_image_label_pairs(
        self,
        image_directory: Path,
        label_directory: Path,
        split_name: str,
    ) -> bool:

        if not self._check_directory(image_directory):
            return False

        if not self._check_directory(label_directory):
            return False

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        images = [
            image
            for image in image_directory.iterdir()
            if image.is_file()
            and image.suffix.lower() in image_extensions
        ]

        if not images:
            print(f"No images found in {split_name} dataset.")
            return False

        is_valid = True

        image_stems = {image.stem for image in images}

        label_files = {
            label.stem
            for label in label_directory.glob("*.txt")
        }

        missing_labels = image_stems - label_files
        extra_labels = label_files - image_stems

        if missing_labels:
            is_valid = False
            print(
                f"\nMissing labels in {split_name}: "
                f"{len(missing_labels)}"
            )

            for stem in sorted(missing_labels)[:10]:
                print(f"  Missing: {stem}.txt")

        if extra_labels:
            is_valid = False
            print(
                f"\nLabels without corresponding images in "
                f"{split_name}: {len(extra_labels)}"
            )

            for stem in sorted(extra_labels)[:10]:
                print(f"  Extra label: {stem}.txt")

        print(
            f"\n{split_name.upper()} DATASET"
        )
        print(f"Images: {len(images)}")
        print(f"Labels: {len(label_files)}")

        return is_valid

    def _validate_yolo_labels(
        self,
        label_directory: Path,
        number_of_classes: int,
        split_name: str,
    ) -> bool:

        is_valid = True

        label_files = list(label_directory.glob("*.txt"))

        invalid_files = 0
        invalid_annotations = 0

        for label_file in label_files:

            try:
                with open(label_file, "r", encoding="utf-8") as file:
                    lines = file.readlines()

                for line_number, line in enumerate(lines, start=1):

                    line = line.strip()

                    if not line:
                        continue

                    values = line.split()

                    if len(values) != 5:
                        print(
                            f"Invalid annotation format: "
                            f"{label_file.name}, "
                            f"line {line_number}"
                        )

                        invalid_annotations += 1
                        is_valid = False
                        continue

                    class_id = int(values[0])
                    coordinates = [
                        float(value)
                        for value in values[1:]
                    ]

                    if not (
                        0 <= class_id < number_of_classes
                    ):
                        print(
                            f"Invalid class ID in "
                            f"{label_file.name}: {class_id}"
                        )

                        invalid_annotations += 1
                        is_valid = False

                    if not all(
                        0.0 <= value <= 1.0
                        for value in coordinates
                    ):
                        print(
                            f"Invalid bounding box values in "
                            f"{label_file.name}, "
                            f"line {line_number}"
                        )

                        invalid_annotations += 1
                        is_valid = False

            except (ValueError, OSError) as error:
                print(
                    f"Could not read label file "
                    f"{label_file.name}: {error}"
                )

                invalid_files += 1
                is_valid = False

        print(
            f"\n{split_name.upper()} LABEL VALIDATION"
        )
        print(f"Label files checked: {len(label_files)}")
        print(f"Invalid files: {invalid_files}")
        print(f"Invalid annotations: {invalid_annotations}")

        return is_valid

    def _validate_images(
        self,
        image_directory: Path,
        split_name: str,
    ) -> bool:

        is_valid = True

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        images = [
            image
            for image in image_directory.iterdir()
            if image.is_file()
            and image.suffix.lower() in image_extensions
        ]

        corrupted_images = 0

        for image_file in images:

            image = cv2.imread(str(image_file))

            if image is None:
                print(
                    f"Corrupted or unreadable image: "
                    f"{image_file.name}"
                )

                corrupted_images += 1
                is_valid = False

        print(
            f"\n{split_name.upper()} IMAGE VALIDATION"
        )
        print(f"Images checked: {len(images)}")
        print(f"Corrupted images: {corrupted_images}")

        return is_valid

    def _validate_data_yaml(self) -> bool:

        if not self.data_yaml.exists():
            print(
                f"data.yaml not found: {self.data_yaml}"
            )
            return False

        try:
            with open(
                self.data_yaml,
                "r",
                encoding="utf-8",
            ) as file:
                data = yaml.safe_load(file)

            if not isinstance(data, dict):
                print("Invalid data.yaml format.")
                return False

            if "names" not in data:
                print(
                    "The 'names' field is missing "
                    "from data.yaml."
                )
                return False

            names = data["names"]

            if isinstance(names, list):
                number_of_classes = len(names)
            elif isinstance(names, dict):
                number_of_classes = len(names)
            else:
                print(
                    "Invalid 'names' format in data.yaml."
                )
                return False

            if number_of_classes == 0:
                print("No classes found in data.yaml.")
                return False

            print("\nDATA.YAML VALIDATION")
            print(f"Number of classes: {number_of_classes}")
            print(f"Classes: {names}")

            return True

        except (OSError, yaml.YAMLError) as error:
            print(
                f"Could not read data.yaml: {error}"
            )
            return False

    def validate_dataset(self) -> bool:

        print("Starting dataset validation...")

        yaml_valid = self._validate_data_yaml()

        if not yaml_valid:
            return False

        with open(
            self.data_yaml,
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file)

        names = data["names"]

        if isinstance(names, list):
            number_of_classes = len(names)
        else:
            number_of_classes = len(names)

        validation_results = []

        splits = [
            (
                "train",
                self.train_images,
                self.train_labels,
            ),
            (
                "valid",
                self.valid_images,
                self.valid_labels,
            ),
            (
                "test",
                self.test_images,
                self.test_labels,
            ),
        ]

        for split_name, image_directory, label_directory in splits:

            pair_valid = self._check_image_label_pairs(
                image_directory,
                label_directory,
                split_name,
            )

            image_valid = self._validate_images(
                image_directory,
                split_name,
            )

            label_valid = self._validate_yolo_labels(
                label_directory,
                number_of_classes,
                split_name,
            )

            validation_results.append(
                pair_valid
                and image_valid
                and label_valid
            )

        dataset_valid = (
            yaml_valid
            and all(validation_results)
        )

        print("\n" + "=" * 50)

        if dataset_valid:
            print("DATASET VALIDATION PASSED")
        else:
            print("DATASET VALIDATION FAILED")

        print("=" * 50)

        return dataset_valid


if __name__ == "__main__":

    dataset_path = (
        r"D:\Object-Detection-Project"
        r"\data\Sign detection.v5i.yolov11"
    )

    data_validation = DataValidation(
        dataset_path=dataset_path
    )

    data_validation.validate_dataset()
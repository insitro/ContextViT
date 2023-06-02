# Get the Camelyon17 dataset from wilds
import os

import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset

from contextvit import transformations

# We use the same split as the official Camelyon17-WILDS repo
# https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/wilds/datasets/camelyon17_dataset.py#L14
TEST_CENTER = 2
VAL_CENTER = 1


class Camelyon17(Dataset):
    def __init__(self, split: str, is_train: bool, transform_cfg: DictConfig):
        """
        Labeled splits: train, id_val (in distribution), val (OOD), test
        Unlabeled splits: train_unlabeled, val_unlabeled, test_unlabeled

        Input (x):
            96x96 image patches extracted from histopathology slides.

        Label (y):
            y is binary. It is 1 if the central 32x32 region contains any tumor tissue, and 0 otherwise.

        Metadata:
            Each patch is annotated with the ID of the hospital it came from (integer from 0 to 4)
            and the slide it came from (integer from 0 to 49).
        """
        super().__init__()
        self.is_train = is_train
        self.transform = getattr(transformations, transform_cfg.name)(
            is_train=is_train, **transform_cfg.args
        )

        if "WILDS_DATA_PATH" not in os.environ:
            raise ValueError("You need to set the enviornment variable WILDS_DATA_PATH")
        else:
            self.base_path = os.environ["WILDS_DATA_PATH"]

        if split == "train":
            self.base_path = os.path.join(self.base_path, "camelyon17_v1.0")
            self.df = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))
            self.df = self.df[self.df["split"] == 0]
            self.df = self.df[self.df["center"] != TEST_CENTER]
            self.df = self.df[self.df["center"] != VAL_CENTER]
        elif split == "id_val":
            self.base_path = os.path.join(self.base_path, "camelyon17_v1.0")
            self.df = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))
            self.df = self.df[self.df["split"] == 1]
            self.df = self.df[self.df["center"] != TEST_CENTER]
            self.df = self.df[self.df["center"] != VAL_CENTER]
        elif split == "test":
            self.base_path = os.path.join(self.base_path, "camelyon17_v1.0")
            self.df = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))
            self.df = self.df[self.df["center"] == TEST_CENTER]
        elif split == "val":
            self.base_path = os.path.join(self.base_path, "camelyon17_v1.0")
            self.df = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))
            self.df = self.df[self.df["center"] == VAL_CENTER]
        elif split == "train_unlabeled":
            self.base_path = os.path.join(self.base_path, "camelyon17_unlabeled_v1.0")
            self.df = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))
            self.df = self.df[self.df["center"] != TEST_CENTER]
            self.df = self.df[self.df["center"] != VAL_CENTER]
        elif split == "val_unlabeled":
            self.base_path = os.path.join(self.base_path, "camelyon17_unlabeled_v1.0")
            self.df = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))
            self.df = self.df[self.df["center"] == VAL_CENTER]
        elif split == "test_unlabeled":
            self.base_path = os.path.join(self.base_path, "camelyon17_unlabeled_v1.0")
            self.df = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))
            self.df = self.df[self.df["center"] == TEST_CENTER]
        elif split == "unlabeled":
            self.base_path = os.path.join(self.base_path, "camelyon17_unlabeled_v1.0")
            self.df = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))
        else:
            raise ValueError(f"Unknown split {split}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(
            self.base_path,
            (
                f"patches/patient_{row.patient:03}_node_{row.node}/"
                f"patch_patient_{row.patient:03}_node_{row.node}_x_{row.x_coord}_y_{row.y_coord}.png"
            ),
        )

        try:
            img_pil = Image.open(path).convert("RGB")
        except Exception as e:
            print(path)
            raise e

        return (
            self.transform(img_pil),
            {
                "tumor": row.tumor,  # 0 or 1 for labeled,  -1 for unlabeled
                "hospital_id": row.center,
                "slide_id": row.slide,
                "node_id": row.node,
                "patient_id": row.patient,
            },
        )

    def __len__(self) -> int:
        return len(self.df)

import io
from typing import List, Union

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config
from omegaconf import DictConfig
from torch.utils.data import Dataset

from contextvit import transformations


def load_meta_data():
    PLATE_TO_ID = {"BR00116991": 0, "BR00116993": 1, "BR00117000": 2}
    FIELD_TO_ID = dict(zip([str(i) for i in range(1, 10)], range(9)))
    WELL_TO_ID = {}
    for i in range(16):
        for j in range(1, 25):
            well_loc = f"{chr(ord('A') + i)}{j:02d}"
            WELL_TO_ID[well_loc] = len(WELL_TO_ID)

    WELL_TO_LBL = {}
    # map the well location to the perturbation label
    # Note that the platemaps are different for different perturbations
    base_path = "s3://insitro-research-2023-context-vit/jumpcp/platemap_and_metadata"
    PLATE_MAP = {
        "compound": f"{base_path}/JUMP-Target-1_compound_platemap.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_platemap.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_platemap.tsv",
    }
    META_DATA = {
        "compound": f"{base_path}/JUMP-Target-1_compound_metadata.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_metadata.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_metadata.tsv",
    }
    for perturbation in PLATE_MAP.keys():
        df_platemap = pd.read_parquet(PLATE_MAP[perturbation])
        df_metadata = pd.read_parquet(META_DATA[perturbation])
        df = df_metadata.merge(df_platemap, how="inner", on="broad_sample")

        if perturbation == "compound":
            target_name = "target"
        else:
            target_name = "gene"

        codes, uniques = pd.factorize(df[target_name])
        codes += 1  # set none (neg control) to id 0
        assert min(codes) == 0
        print(f"{target_name} has {len(uniques)} unique values")
        WELL_TO_LBL[perturbation] = dict(zip(df["well_position"], codes))

    return PLATE_TO_ID, FIELD_TO_ID, WELL_TO_ID, WELL_TO_LBL


class JUMPCP(Dataset):
    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None

    def __init__(
        self,
        cyto_mask_path_list: str,
        is_train: bool,
        split: str,
        transform_cfg: DictConfig,
        perturbation_type: str = "",
    ) -> None:
        """Initialize the dataset."""
        super().__init__()

        self.s3_client = boto3.client(
            "s3", config=Config(retries=dict(max_attempts=10))
        )

        # read the cyto mask df
        df = pd.concat(
            [pd.read_parquet(path) for path in cyto_mask_path_list], ignore_index=True
        )
        if split != "all":
            np.random.seed(0)
            perm = np.random.permutation(df.index)
            if split == "train":
                df = df.iloc[perm[: len(df) // 2]]
            elif split == "test":
                df = df.iloc[perm[len(df) // 2 :]]
            else:
                raise ValueError(f"Unknown split {split}")

        self.data_path = list(df["path"])
        self.data_id = list(df["ID"])
        self.well_loc = list(df["well_loc"])

        self.perturbation_type = perturbation_type
        self.plate2id, self.field2id, self.well2id, self.well2lbl = load_meta_data()

        self.transform = getattr(transformations, transform_cfg.name)(
            is_train,
            **transform_cfg.args,
            normalization_mean=transform_cfg.normalization.mean,
            normalization_std=transform_cfg.normalization.std,
        )

    def __getitem__(self, index):

        img_chw = self.get_image(self.data_path[index])
        if img_chw is None:
            return None

        img_hwc = img_chw.transpose(1, 2, 0)

        if self.perturbation_type == "":
            label = -1
        else:
            label = self.well2lbl[self.perturbation_type][self.well_loc[index]]

        return (
            self.transform(img_hwc),
            {
                "label": label,
                "well_id": self.well2id[self.well_loc[index]],
                "plate_id": self.plate2id[self.data_id[index].split("_")[0]],
                "field_id": self.field2id[self.data_id[index].split("_")[2]],
            },
        )

    def __len__(self) -> int:
        return len(self.data_path)

    def get_image(self, path, max_attempts=10):
        bucket, key = path.replace("s3://", "", 1).split("/", 1)
        while True:
            try:
                s3_obj = self.s3_client.get_object(Bucket=bucket, Key=key)
                out = np.load(io.BytesIO(s3_obj["Body"].read()), allow_pickle=True)
                return out

            except Exception as e:
                if max_attempts > 0:
                    max_attempts -= 1
                else:
                    raise e

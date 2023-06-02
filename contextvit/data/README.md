# Setup datasets
We use the WILDS package (included in the contextvit conda environment) to download the dataset.
First we need to clone the WILD
### Cloning the W
However, it is still required to 


```bash
# clone the wilds repo
git clone git@github.com:p-lambda/wilds.git
cd wilds

# Download the labeled data for Camelyon17, iWildCam and FMoW.
python wilds/download_datasets.py --root_dir data --datasets camelyon17
python wilds/download_datasets.py --root_dir data --datasets fmow
python wilds/download_datasets.py --root_dir data --datasets iwildcam

# Download the unlabeled data for Camelyon17
python wilds/download_datasets.py --root_dir data --datasets camelyon17 --unlabeled True
```


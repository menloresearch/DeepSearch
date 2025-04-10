"""Download and extract FlashRAG index."""

import os
import zipfile

import requests
from tqdm import tqdm

from config import DATA_DIR

# Constants
URL = "https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/resolve/master/retrieval_corpus/wiki18_100w_e5_index.zip"
ZIP_NAME = "wiki18_100w_e5_index.zip"
zip_path = DATA_DIR / ZIP_NAME

# Download with progress bar
print("📥 Downloading index...")
response = requests.get(URL, stream=True)
total_size = int(response.headers.get("content-length", 0))

with (
    open(zip_path, "wb") as f,
    tqdm(
        desc=ZIP_NAME,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar,
):
    for data in response.iter_content(chunk_size=1024):
        size = f.write(data)
        bar.update(size)

# Extract
print("📦 Extracting index...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)

# Clean up zip
os.remove(zip_path)
print("✅ Download and extraction completed successfully!")
print(f"Index file is at: {DATA_DIR}/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/e5_flat_inner.index")

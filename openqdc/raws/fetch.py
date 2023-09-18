"""Script to download the molecule3d dataset from Google Drive."""
import os
import tqdm
import gdown
import fsspec
import socket
import tarfile
import zipfile
import requests
import urllib.error
import urllib.request
from loguru import logger
from sklearn.utils import Bunch
from openqdc.utils.paths import get_local_cache
from openqdc.raws.config_factory import DataConfigFactory


# function to download large files with requests
def fetch_file(url, local_filename, overwrite=False):
    """
    Download a file from a url to a local file.
    Parameters
    ----------
    url : str
        URL to download from.
    local_filename : str
        Local file to save to.
    overwrite : bool
        Whether to overwrite existing files.
    Returns
    -------
    local_filename : str
        Local file.
    """
    try:

        if os.path.exists(local_filename) and not overwrite:
            logger.info("File already exists, skipping download")
        else:
            logger.info(f"File: {local_filename}")
            if "drive.google.com" in url:
                gdown.download(url, local_filename, quiet=False)
            else:
                r = requests.get(url, stream=True)
                with fsspec.open(local_filename, "wb") as f:
                    for chunk in tqdm.tqdm(r.iter_content(chunk_size=16384)):
                        if chunk:
                            f.write(chunk)

        # decompress archive if necessary
        parent = os.path.dirname(local_filename)
        if local_filename.endswith("tar.gz"):            
            with tarfile.open(local_filename) as tar:
                logger.info(f"Verifying archive extraction states: {local_filename}")
                all_names = tar.getnames()
                all_extracted = all([os.path.exists(os.path.join(parent, x)) for x in all_names])
                if not all_extracted:
                    logger.info(f"Extracting archive: {local_filename}")
                    tar.extractall(path=parent)
                else:
                    logger.info(f"Archive already extracted: {local_filename}")

        elif local_filename.endswith("zip"):
            logger.info(f"Verifying archive extraction states: {local_filename}")
            with zipfile.ZipFile(local_filename, "r") as zip_ref:
                all_names = zip_ref.namelist()
                all_extracted = all([os.path.exists(os.path.join(parent, x)) for x in all_names])
                if not all_extracted:
                    logger.info(f"Extracting archive: {local_filename}")
                    zip_ref.extractall(parent)
                else:
                    logger.info(f"Archive already extracted: {local_filename}")

        elif local_filename.endswith("xz"):
            logger.info(f"Excloabout:blanktracting archive: {local_filename}")

            os.system(f"cd {parent} && xz -d *.xz")
        else:
            pass

    except (socket.gaierror, urllib.error.URLError) as err:
        raise ConnectionError("Could not download {} due to {}".format(url, err))

    return local_filename


class DataDownloader:
    """Download data from a remote source.
    Parameters
    ----------
    cache_path : str
        Path to the cache directory.
    overwrite : bool
        Whether to overwrite existing files.
    """

    def __init__(self, cache_path=None, overwrite=False):
        if cache_path is None:
            cache_path = get_local_cache()

        self.cache_path = cache_path
        self.overwrite = overwrite
    
    def from_config(self, config: dict):
        b_config = Bunch(**config)
        data_path = os.path.join(self.cache_path, b_config.dataset_name)
        os.makedirs(data_path, exist_ok=True)

        logger.info(f"Downloading the {b_config.dataset_name} dataset")
        for local, link in b_config.links.items():
            outfile = os.path.join(data_path, local) 

            fetch_file(link, outfile)

    def from_name(self, name):
        cfg = DataConfigFactory()(name)
        return self.from_config(cfg)
 

if __name__ == "__main__":
    for dataset_name in DataConfigFactory.available_datasets:
        dd = DataDownloader()
        dd.from_name(dataset_name)


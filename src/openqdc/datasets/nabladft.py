import os
from os.path import join as p_join
from typing import Dict

import datamol as dm
import numpy as np
from nablaDFT.dataset import HamiltonianDatabase
from tqdm import tqdm

from openqdc.datasets.base import BaseDataset


def to_mol(entry) -> Dict[str, np.ndarray]:
    Z, R, E, F = entry[:4]
    C = np.zeros_like(Z)

    res = dict(
        atomic_inputs=np.concatenate((Z[:, None], C[:, None], R), axis=-1).astype(np.float32),
        name=np.array([""]),
        energies=E[:, None].astype(np.float32),
        forces=F[:, :, None].astype(np.float32),
        n_atoms=np.array([Z.shape[0]], dtype=np.int32),
        subset=np.array(["nabla"]),
    )

    return res


def read_chunk_from_db(raw_path, start_idx, stop_idx, step_size=1000):
    print(f"Loading from {start_idx} to {stop_idx}")
    db = HamiltonianDatabase(raw_path)
    idxs = list(np.arange(start_idx, stop_idx))
    n, s = len(idxs), step_size

    samples = [to_mol(entry) for i in tqdm(range(0, n, s)) for entry in db[idxs[i : i + s]]]
    return samples


class NablaDFT(BaseDataset):
    """
    NablaDFT is a dataset constructed from a subset of the
    [Molecular Sets (MOSES) dataset](https://github.com/molecularsets/moses) consisting of 1 million molecules
    with 5,340,152 unique conformations generated using ωB97X-D/def2-SVP level of theory.

    Usage:
    ```python
    from openqdc.datasets import NablaDFT
    dataset = NablaDFT()
    ```

    References:
    - https://pubs.rsc.org/en/content/articlelanding/2022/CP/D2CP03966D
    - https://github.com/AIRI-Institute/nablaDFT
    """

    __name__ = "nabladft"
    __energy_methods__ = ["wb97x-d/def2-svp"]

    energy_target_names = ["wb97x-d/def2-svp"]
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"

    def __init__(self, energy_unit=None, distance_unit=None) -> None:
        super().__init__(energy_unit=energy_unit, distance_unit=distance_unit)

    def read_raw_entries(self):
        raw_path = p_join(self.root, "dataset_full.db")
        train = HamiltonianDatabase(raw_path)
        n, c = len(train), 20
        step_size = int(np.ceil(n / os.cpu_count()))

        fn = lambda i: read_chunk_from_db(raw_path, i * step_size, min((i + 1) * step_size, n))
        samples = dm.parallelized(
            fn, list(range(c)), n_jobs=c, progress=False, scheduler="threads"
        )  # don't use more than 1 job

        return sum(samples, [])

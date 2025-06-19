"""The BaseDataset defining shared functionality between all datasets."""

import os
from functools import partial
from itertools import compress
from os.path import join as p_join
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from ase import Atoms
from ase.io.extxyz import write_extxyz
from loguru import logger
from sklearn.utils import Bunch
from tqdm import tqdm

from openqdc.datasets.energies import AtomEnergies
from openqdc.datasets.properties import DatasetPropertyMixIn
from openqdc.datasets.statistics import (
    ForcesCalculatorStats,
    FormationEnergyStats,
    PerAtomFormationEnergyStats,
    StatisticManager,
    TotalEnergyStats,
)
from openqdc.datasets.structure import MemMapDataset, ZarrDataset
from openqdc.utils.constants import MAX_CHARGE, NB_ATOMIC_FEATURES
from openqdc.utils.descriptors import get_descriptor
from openqdc.utils.exceptions import (
    DatasetNotAvailableError,
    StatisticsNotAvailableError,
)
from openqdc.utils.io import (
    copy_exists,
    dict_to_atoms,
    get_local_cache,
    push_remote,
    set_cache_dir,
)
from openqdc.utils.package_utils import has_package, requires_package
from openqdc.utils.regressor import Regressor  # noqa
from openqdc.utils.units import (
    DistanceTypeConversion,
    EnergyTypeConversion,
    ForceTypeConversion,
    get_conversion,
)

if has_package("torch"):
    import torch

if has_package("jax"):
    import jax.numpy as jnp


@requires_package("torch")
def to_torch(x: np.ndarray):
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x)


@requires_package("jax")
def to_jax(x: np.ndarray):
    if isinstance(x, jnp.ndarray):
        return x
    return jnp.array(x)


_CONVERT_DICT = {"torch": to_torch, "jax": to_jax, "numpy": lambda x: x}


class BaseDataset(DatasetPropertyMixIn):
    """
    Base class for datasets in the openQDC package.
    """

    energy_target_names = []
    force_target_names = []
    read_as_zarr = False
    __energy_methods__ = []
    __force_mask__ = []
    __isolated_atom_energies__ = []
    _fn_energy = lambda x: x
    _fn_distance = lambda x: x
    _fn_forces = lambda x: x

    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    __average_nb_atoms__ = None
    __links__ = {}

    def __init__(
        self,
        energy_unit: Optional[str] = None,
        distance_unit: Optional[str] = None,
        array_format: str = "numpy",
        energy_type: Optional[str] = "formation",
        overwrite_local_cache: bool = False,
        cache_dir: Optional[str] = None,
        recompute_statistics: bool = False,
        transform: Optional[Callable] = None,
        skip_statistics: bool = False,
        read_as_zarr: bool = False,
        regressor_kwargs: Dict = {
            "solver_type": "linear",
            "sub_sample": None,
            "stride": 1,
        },
    ) -> None:
        """

        Parameters:
            energy_unit:
                Energy unit to convert dataset to. Supported units: ["kcal/mol", "kj/mol", "hartree", "ev"]
            distance_unit:
                Distance unit to convert dataset to. Supported units: ["ang", "nm", "bohr"]
            array_format:
                Format to return arrays in. Supported formats: ["numpy", "torch", "jax"]
            energy_type:
                Type of isolated atom energy to use for the dataset. Default: "formation"
                Supported types: ["formation", "regression", "null", None]
            overwrite_local_cache:
                Whether to overwrite the locally cached dataset.
            cache_dir:
                Cache directory location. Defaults to "~/.cache/openqdc"
            recompute_statistics:
                Whether to recompute the statistics of the dataset.
            transform:
                transformation to apply to the __getitem__ calls
            regressor_kwargs:
                Dictionary of keyword arguments to pass to the regressor.
                Default: {"solver_type": "linear", "sub_sample": None, "stride": 1}
                solver_type can be one of ["linear", "ridge"]
        """
        set_cache_dir(cache_dir)
        # self._init_lambda_fn()
        self.data = None
        self._original_unit = self.energy_unit
        self.recompute_statistics = recompute_statistics
        self.regressor_kwargs = regressor_kwargs
        self.transform = transform
        self.read_as_zarr = read_as_zarr
        self.energy_type = energy_type if energy_type is not None else "null"
        self.refit_e0s = recompute_statistics or overwrite_local_cache
        self.skip_statistics = skip_statistics
        if not self.is_preprocessed():
            raise DatasetNotAvailableError(self.__name__)
        else:
            self.read_preprocess(overwrite_local_cache=overwrite_local_cache)
        self.set_array_format(array_format)
        self._post_init(overwrite_local_cache, energy_unit, distance_unit)

    def _init_lambda_fn(self):
        self._fn_energy = lambda x: x
        self._fn_distance = lambda x: x
        self._fn_forces = lambda x: x

    @property
    def dataset_wrapper(self):
        if not hasattr(self, "_dataset_wrapper"):
            self._dataset_wrapper = ZarrDataset() if self.read_as_zarr else MemMapDataset()
        return self._dataset_wrapper

    @property
    def config(self):
        assert len(self.__links__) > 0, "No links provided for fetching"
        return dict(dataset_name=self.__name__, links=self.__links__)

    @classmethod
    def fetch(cls, cache_path: Optional[str] = None, overwrite: bool = False) -> None:
        from openqdc.utils.download_api import DataDownloader

        DataDownloader(cache_path, overwrite).from_config(cls.no_init().config)

    def _post_init(
        self,
        overwrite_local_cache: bool = False,
        energy_unit: Optional[str] = None,
        distance_unit: Optional[str] = None,
    ) -> None:
        self._set_units(None, None)
        self._set_isolated_atom_energies()
        if not self.skip_statistics:
            self._precompute_statistics(overwrite_local_cache=overwrite_local_cache)
        self._set_units(energy_unit, distance_unit)
        self._convert_data()
        self._set_isolated_atom_energies()

    def _precompute_statistics(self, overwrite_local_cache: bool = False):
        # if self.recompute_statistics or overwrite_local_cache:
        self.statistics = StatisticManager(
            self,
            self.recompute_statistics or overwrite_local_cache,  # check if we need to recompute
            # Add the common statistics (Forces, TotalE, FormE, PerAtomE)
            ForcesCalculatorStats,
            TotalEnergyStats,
            FormationEnergyStats,
            PerAtomFormationEnergyStats,
        )
        self.statistics.run_calculators()  # run the calculators
        self._compute_average_nb_atoms()

    @classmethod
    def no_init(cls):
        """
        Class method to avoid the __init__ method to be called when the class is instanciated.
        Useful for debugging purposes or preprocessing data.
        """
        return cls.__new__(cls)

    @property
    def __force_methods__(self):
        """
        For backward compatibility. To be removed in the future.
        """
        return self.force_methods

    @property
    def energy_methods(self) -> List[str]:
        """Return the string version of the energy methods"""
        return [str(i) for i in self.__energy_methods__]

    @property
    def force_mask(self):
        if len(self.__class__.__force_mask__) == 0:
            self.__class__.__force_mask__ = [False] * len(self.__energy_methods__)
        return self.__class__.__force_mask__

    @property
    def force_methods(self):
        return list(compress(self.energy_methods, self.force_mask))

    @property
    def e0s_dispatcher(self) -> AtomEnergies:
        """
        Property to get the object that dispatched the isolated atom energies of the QM methods.

        Returns:
            Object wrapping the isolated atom energies of the QM methods.
        """
        if not hasattr(self, "_e0s_dispatcher"):
            # Automatically fetch/compute formation or regression energies
            self._e0s_dispatcher = AtomEnergies(self, **self.regressor_kwargs)
        return self._e0s_dispatcher

    def _convert_data(self):
        logger.info(
            f"Converting {self.__name__} data to the following units:\n\
                     Energy: {str(self.energy_unit)},\n\
                     Distance: {str(self.distance_unit)},\n\
                     Forces: {str(self.force_unit) if self.__force_methods__ else 'None'}"
        )
        for key in self.data_keys:
            self.data[key] = self._convert_on_loading(self.data[key], key)

    @property
    def energy_unit(self):
        return EnergyTypeConversion(self.__energy_unit__)

    @property
    def distance_unit(self):
        return DistanceTypeConversion(self.__distance_unit__)

    @property
    def force_unit(self):
        units = self.__forces_unit__.split("/")
        if len(units) > 2:
            units = ["/".join(units[:2]), units[-1]]
        return ForceTypeConversion(tuple(units))  # < 3.12 compatibility

    @property
    def root(self):
        return p_join(get_local_cache(), self.__name__)

    @property
    def preprocess_path(self):
        path = p_join(self.root, "preprocessed")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def data_keys(self):
        keys = list(self.data_types.keys())
        if len(self.__force_methods__) == 0:
            keys.remove("forces")
        return keys

    @property
    def pkl_data_keys(self):
        return list(self.pkl_data_types.keys())

    @property
    def pkl_data_types(self):
        return {"name": str, "subset": str, "n_atoms": np.int32}

    @property
    def atom_energies(self):
        return self._e0s_dispatcher

    @property
    def data_types(self):
        return {
            "atomic_inputs": np.float32,
            "position_idx_range": np.int32,
            "energies": np.float64,
            "forces": np.float32,
        }

    @property
    def data_shapes(self):
        return {
            "atomic_inputs": (-1, NB_ATOMIC_FEATURES),
            "position_idx_range": (-1, 2),
            "energies": (-1, len(self.energy_methods)),
            "forces": (-1, 3, len(self.force_methods)),
        }

    def _set_units(self, en: Optional[str] = None, ds: Optional[str] = None):
        old_en, old_ds = self.energy_unit, self.distance_unit
        en = en if en is not None else old_en
        ds = ds if ds is not None else old_ds
        self.set_energy_unit(en)
        self.set_distance_unit(ds)
        if self.__force_methods__:
            self._fn_forces = self.force_unit.to(str(self.energy_unit), str(self.distance_unit))
            self.__forces_unit__ = str(self.energy_unit) + "/" + str(self.distance_unit)

    def _set_isolated_atom_energies(self):
        if self.__energy_methods__ is None:
            logger.error("No energy methods defined for this dataset.")
        if self.energy_type == "formation":
            f = get_conversion("hartree", self.__energy_unit__)
        else:
            # regression are calculated on the original unit of the dataset
            f = self._original_unit.to(self.energy_unit)
        self.__isolated_atom_energies__ = f(self.e0s_dispatcher.e0s_matrix)

    def convert_energy(self, x):
        return self._fn_energy(x)

    def convert_distance(self, x):
        return self._fn_distance(x)

    def convert_forces(self, x):
        return self._fn_forces(x)

    def set_energy_unit(self, value: str):
        """
        Set a new energy unit for the dataset.

        Parameters:
            value:
                New energy unit to set.
        """
        # old_unit = self.energy_unit
        # self.__energy_unit__ = value
        self._fn_energy = self.energy_unit.to(value)  # get_conversion(old_unit, value)
        self.__energy_unit__ = value

    def set_distance_unit(self, value: str):
        """
        Set a new distance unit for the dataset.

        Parameters:
            value:
                New distance unit to set.
        """
        # old_unit = self.distance_unit
        # self.__distance_unit__ = value
        self._fn_distance = self.distance_unit.to(value)  # get_conversion(old_unit, value)
        self.__distance_unit__ = value

    def set_array_format(self, format: str):
        assert format in ["numpy", "torch", "jax"], f"Format {format} not supported."
        self.array_format = format

    def read_raw_entries(self):
        """
        Preprocess the raw (aka from the fetched source) into a list of dictionaries.
        """
        raise NotImplementedError

    def collate_list(self, list_entries: List[Dict]) -> Dict:
        """
        Collate a list of entries into a single dictionary.

        Parameters:
            list_entries:
                List of dictionaries containing the entries to collate.

        Returns:
            Dictionary containing the collated entries.
        """
        # concatenate entries
        res = {key: np.concatenate([r[key] for r in list_entries if r is not None], axis=0) for key in list_entries[0]}

        csum = np.cumsum(res.get("n_atoms"))
        x = np.zeros((csum.shape[0], 2), dtype=np.int32)
        x[1:, 0], x[:, 1] = csum[:-1], csum
        res["position_idx_range"] = x

        return res

    def save_preprocess(
        self, data_dict: Dict[str, np.ndarray], upload: bool = False, overwrite: bool = True, as_zarr: bool = False
    ):
        """
        Save the preprocessed data to the cache directory and optionally upload it to the remote storage.

        Parameters:
            data_dict:
                Dictionary containing the preprocessed data.
            upload:
                Whether to upload the preprocessed data to the remote storage or only saving it locally.
            overwrite:
                Whether to overwrite the preprocessed data if it already exists.
                Only used if upload is True. Cache is always overwritten locally.
        """
        # save memmaps
        logger.info("Preprocessing data and saving it to cache.")
        paths = self.dataset_wrapper.save_preprocess(
            self.preprocess_path, self.data_keys, data_dict, self.pkl_data_keys, self.pkl_data_types
        )
        if upload:
            for local_path in paths:
                push_remote(local_path, overwrite=overwrite)  # make it async?

    def read_preprocess(self, overwrite_local_cache=False):
        logger.info("Reading preprocessed data.")
        logger.info(
            f"Dataset {self.__name__} with the following units:\n\
                     Energy: {self.energy_unit},\n\
                     Distance: {self.distance_unit},\n\
                     Forces: {self.force_unit if self.force_methods else 'None'}"
        )

        self.data = self.dataset_wrapper.load_data(
            self.preprocess_path,
            self.data_keys,
            self.data_types,
            self.data_shapes,
            self.pkl_data_keys,
            overwrite_local_cache,
        )  # this should be async if possible
        for key in self.data:
            logger.info(f"Loaded {key} with shape {self.data[key].shape}, dtype {self.data[key].dtype}")

    def _convert_on_loading(self, x, key):
        if key == "energies":
            return self.convert_energy(x)
        elif key == "forces":
            return self.convert_forces(x)
        elif key == "atomic_inputs":
            x = np.array(x, dtype=np.float32)
            x[:, -3:] = self.convert_distance(x[:, -3:])
            return x
        else:
            return x

    def is_preprocessed(self) -> bool:
        """
        Check if the dataset is preprocessed and available online or locally.

        Returns:
            True if the dataset is available remotely or locally, False otherwise.
        """
        predicats = [
            copy_exists(p_join(self.preprocess_path, self.dataset_wrapper.add_extension(f"{key}")))
            for key in self.data_keys
        ]
        predicats += [copy_exists(p_join(self.preprocess_path, file)) for file in self.dataset_wrapper._extra_files]
        return all(predicats)

    def is_cached(self) -> bool:
        """
        Check if the dataset is cached locally.

        Returns:
            True if the dataset is cached locally, False otherwise.
        """
        predicats = [
            os.path.exists(p_join(self.preprocess_path, self.dataset_wrapper.add_extension(f"{key}")))
            for key in self.data_keys
        ]
        predicats += [copy_exists(p_join(self.preprocess_path, file)) for file in self.dataset_wrapper._extra_files]
        return all(predicats)

    def preprocess(self, upload: bool = False, overwrite: bool = True, as_zarr: bool = True):
        """
        Preprocess the dataset and save it.

        Parameters:
            upload:
                Whether to upload the preprocessed data to the remote storage or only saving it locally.
            overwrite:
                hether to overwrite the preprocessed data if it already exists.
                Only used if upload is True. Cache is always overwritten locally.
            as_zarr:
                Whether to save the data as zarr files
        """
        if overwrite or not self.is_preprocessed():
            entries = self.read_raw_entries()
            res = self.collate_list(entries)
            self.save_preprocess(res, upload, overwrite, as_zarr)

    def upload(self, overwrite: bool = False, as_zarr: bool = False):
        """
        Upload the preprocessed data to the remote storage. Must be called after preprocess and
        need to have write privileges.

        Parameters:
            overwrite:
                Whether to overwrite the remote data if it already exists
            as_zarr:
                Whether to upload the data as zarr files
        """
        for key in self.data_keys:
            local_path = p_join(self.preprocess_path, f"{key}.mmap" if not as_zarr else f"{key}.zip")
            push_remote(local_path, overwrite=overwrite)
        local_path = p_join(self.preprocess_path, "props.pkl" if not as_zarr else "metadata.zip")
        push_remote(local_path, overwrite=overwrite)

    def save_xyz(self, idx: int, energy_method: int = 0, path: Optional[str] = None, ext: bool = True):
        """
        Save a single entry at index idx as an extxyz file.

        Parameters:
            idx:
                Index of the entry
            energy_method:
                Index of the energy method to use
            path:
                Path to save the xyz file. If None, the current working directory is used.
            ext:
                Whether to include additional informations like forces and other metadatas (extxyz format)
        """
        if path is None:
            path = os.getcwd()
        at = self.get_ase_atoms(idx, ext=ext, energy_method=energy_method)
        write_extxyz(p_join(path, f"mol_{idx}.xyz"), at, plain=not ext)

    def to_xyz(self, energy_method: int = 0, path: Optional[str] = None):
        """
        Save dataset as single xyz file (extended xyz format).

        Parameters:
            energy_method:
                Index of the energy method to use
            path:
                Path to save the xyz file
        """
        with open(p_join(path if path else os.getcwd(), f"{self.__name__}.xyz"), "w") as f:
            for atoms in tqdm(
                self.as_iter(atoms=True, energy_method=energy_method),
                total=len(self),
                desc=f"Saving {self.__name__} as xyz file",
            ):
                write_extxyz(f, atoms, append=True)

    def get_ase_atoms(self, idx: int, energy_method: int = 0, ext: bool = True) -> Atoms:
        """
        Get the ASE atoms object for the entry at index idx.

        Parameters:
            idx:
                Index of the entry.
            energy_method:
                Index of the energy method to use
            ext:
                Whether to include additional informations

        Returns:
            ASE atoms object
        """
        entry = self[idx]
        at = dict_to_atoms(entry, ext=ext, energy_method=energy_method)
        return at

    def subsample(
        self, n_samples: Optional[Union[List[int], int, float]] = None, replace: bool = False, seed: int = 42
    ):
        np.random.seed(seed)
        if n_samples is None:
            return list(range(len(self)))
        try:
            if 0 < n_samples < 1:
                n_samples = int(n_samples * len(self))
            if isinstance(n_samples, int):
                idxs = np.random.choice(len(self), size=n_samples, replace=replace)
        except (ValueError, TypeError):  # list, set, np.ndarray
            idxs = n_samples
        return idxs

    @requires_package("datamol")
    def calculate_descriptors(
        self,
        descriptor_name: str = "soap",
        chemical_species: Optional[List[str]] = None,
        n_samples: Optional[Union[List[int], int, float]] = None,
        progress: bool = True,
        **descriptor_kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Compute the descriptors for the dataset.

        Parameters:
            descriptor_name:
                Name of the descriptor to use. Supported descriptors are ["soap"]
            chemical_species:
                List of chemical species to use for the descriptor computation, by default None.
                If None, the chemical species of the dataset are used.
            n_samples:
                Number of samples to use for the computation, by default None.
                If None, all the dataset is used.
                If a list of integers is provided, the descriptors are computed for
                each of the specified idx of samples.
            progress:
                Whether to show a progress bar, by default True.
            **descriptor_kwargs : dict
                Keyword arguments to pass to the descriptor instantiation of the model.

        Returns:
            Dictionary containing the following keys:
                - values : np.ndarray of shape (N, M) containing the descriptors for the dataset
                - idxs : np.ndarray of shape (N,) containing the indices of the samples used

        """
        import datamol as dm

        datum = {}
        idxs = self.subsample(n_samples)
        model = get_descriptor(descriptor_name.lower())(
            species=self.chemical_species if chemical_species is None else chemical_species, **descriptor_kwargs
        )

        def wrapper(idx):
            entry = self.get_ase_atoms(idx, ext=False)
            return model.calculate(entry)

        descr = dm.parallelized(wrapper, idxs, progress=progress, scheduler="threads", n_jobs=-1)
        datum["values"] = np.vstack(descr)
        datum["idxs"] = idxs
        return datum

    def as_iter(self, atoms: bool = False, energy_method: int = 0) -> Iterable:
        """
        Return the dataset as an iterator.

        Parameters:
            atoms:
                Whether to return the items as ASE atoms object, by default False
            energy_method:
                Index of the energy method to use

        Returns:
            Iterator of the dataset
        """

        func = partial(self.get_ase_atoms, energy_method=energy_method) if atoms else self.__getitem__

        for i in range(len(self)):
            yield func(i)

    def __iter__(self):
        for idxs in range(len(self)):
            yield self[idxs]

    def get_statistics(self, return_none: bool = True) -> Dict:
        """
        Get the converted statistics of the dataset.

        Parameters:
            return_none :
                Whether to return None if the statistics for the forces are not available, by default True
                Otherwise, the statistics for the forces are set to 0.0

        Returns:
            Dictionary containing the statistics of the dataset
        """
        selected_stats = self.statistics.get_results()
        if len(selected_stats) == 0:
            raise StatisticsNotAvailableError(self.__name__)
        if not return_none:
            selected_stats.update(
                {
                    "ForcesCalculatorStats": {
                        "mean": np.array([0.0]),
                        "std": np.array([0.0]),
                        "component_mean": np.array([[0.0], [0.0], [0.0]]),
                        "component_std": np.array([[0.0], [0.0], [0.0]]),
                        "component_rms": np.array([[0.0], [0.0], [0.0]]),
                    }
                }
            )
        # cycle trough dict to convert units
        for key, result in selected_stats.items():
            if isinstance(result, ForcesCalculatorStats):
                result.transform(self.convert_forces)
            else:
                result.transform(self.convert_energy)
            result.transform(self._convert_array)
        return {k: result.to_dict() for k, result in selected_stats.items()}

    def __str__(self):
        return f"{self.__name__}"

    def __repr__(self):
        return f"{self.__name__}"

    def __len__(self):
        return self.data["energies"].shape[0]

    def __smiles_converter__(self, x):
        """util function to convert string to smiles: useful if the smiles is
        encoded in a different format than its display format
        """
        return x

    def _convert_array(self, x: np.ndarray):
        return _CONVERT_DICT.get(self.array_format)(x)

    def __getitem__(self, idx: int):
        shift = MAX_CHARGE
        p_start, p_end = self.data["position_idx_range"][idx]
        input = self.data["atomic_inputs"][p_start:p_end]
        z, c, positions, energies = (
            self._convert_array(np.array(input[:, 0], dtype=np.int32)),
            self._convert_array(np.array(input[:, 1], dtype=np.int32)),
            self._convert_array(np.array(input[:, -3:], dtype=np.float32)),
            self._convert_array(np.array(self.data["energies"][idx], dtype=np.float64)),
        )
        name = self.__smiles_converter__(self.data["name"][idx])
        subset = self.data["subset"][idx]
        e0s = self._convert_array(self.__isolated_atom_energies__[..., z, c + shift].T)
        formation_energies = energies - e0s.sum(axis=0)
        forces = None
        if "forces" in self.data:
            forces = self._convert_array(np.array(self.data["forces"][p_start:p_end], dtype=np.float32))

        bunch = Bunch(
            positions=positions,
            atomic_numbers=z,
            charges=c,
            e0=e0s,
            energies=energies,
            formation_energies=formation_energies,
            per_atom_formation_energies=formation_energies / len(z),
            name=name,
            subset=subset,
            forces=forces,
        )

        if self.transform is not None:
            bunch = self.transform(bunch)

        return bunch

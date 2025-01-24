import numpy as np 
from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod

def shape_atom_inputs(coords, atom_species):
    reshaped_coords = coords.reshape(-1, 3)
    frame, atoms, _ = coords.shape
    z = np.tile(atom_species, frame)
    xs = np.stack((z, np.zeros_like(z)), axis=-1)
    return np.concatenate((xs, reshaped_coords), axis=-1, dtype=np.float32)


def read_npz_entry(folder):
    data, name = create_path(folder)
    data = np.load(data)

    nuclear_charges, coords, energies, forces = (
        data["nuclear_charges"],
        data["coords"],
        data["energies"],
        data["forces"],
    )
    frames = coords.shape[0]
    res = dict(
        name=np.array([name] * frames),
        subset=np.array(["dipeptides"] * frames),
        energies=energies[:, None].astype(np.float32),
        forces=forces.reshape(-1, 3, 1).astype(np.float32),
        atomic_inputs=shape_atom_inputs(coords, nuclear_charges),
        n_atoms=np.array([len(nuclear_charges)] * frames, dtype=np.int32),
    )
    return res


def create_path(folder):
    name = folder.split("/")[-1]
    return folder, name 

folder="/network/scratch/s/semih.canturk/cache/openqdc/dipeptides/npz_files/mol_73.npz"

trajectories={
    "mol_73": folder
}

class Dipeptides(BaseDataset):
    """
    """

    __name__ = "dipeptides"

    __energy_methods__ = [PotentialMethod.WB97M_D3BJ_DEF2_TZVPPD]

    energy_target_names = [
        "",
    ]

    __energy_unit__ = "kj/mol"
    __distance_unit__ = "ang"
    __forces_unit__ = "kj/mol/ang"
    
    __force_mask__ = [False]

    @property
    def data_types(self):
        return {
            "atomic_inputs": np.float32,
            "position_idx_range": np.int32,
            "energies": np.float32,
            "forces": np.float32,
        }

    def read_raw_entries(self):
        entries_list = []
        
        for dummy_name, path_to_npz in trajectories.items():
            entries_list.append(read_npz_entry(path_to_npz))
        return entries_list
    

# to store it in the cache and loading back (add the dataset in the __init__)
# Dipeptides.no_init().preprocess(upload=False, overwrite=True)
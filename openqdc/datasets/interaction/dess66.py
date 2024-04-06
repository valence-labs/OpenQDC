import os
from typing import Dict, List

import pandas as pd
from loguru import logger
from tqdm import tqdm

from openqdc.datasets.interaction.base import BaseInteractionDataset
from openqdc.datasets.interaction.des370k import convert_to_record, parse_des_df
from openqdc.methods import InteractionMethod, InterEnergyType

CSV_NAME = {
    "des_s66": "DESS66.csv",
    "des_s66x8": "DESS66x8.csv",
}


class DESS66(BaseInteractionDataset):
    """
    DE Shaw Research interaction energy
    estimates of all 66 conformers from
    the original S66 dataset as described
    in the paper:

    Quantum chemical benchmark databases of gold-standard dimer interaction energies.
    Donchev, A.G., Taube, A.G., Decolvenaere, E. et al.
    Sci Data 8, 55 (2021).
    https://doi.org/10.1038/s41597-021-00833-x

    Data was downloaded from Zenodo:
    https://zenodo.org/records/5676284
    """

    __name__ = "des_s66"
    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "ang"
    __forces_unit__ = "kcal/mol/ang"
    __energy_methods__ = [
        InteractionMethod.MP2_CC_PVDZ,
        InteractionMethod.MP2_CC_PVQZ,
        InteractionMethod.MP2_CC_PVTZ,
        InteractionMethod.MP2_CBS,
        InteractionMethod.CCSD_T_CC_PVDZ,
        InteractionMethod.CCSD_T_CBS,
        InteractionMethod.CCSD_T_NN,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
    ]

    __energy_type__ = [
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.ES,
        InterEnergyType.EX,
        InterEnergyType.EX_S2,
        InterEnergyType.IND,
        InterEnergyType.EX_IND,
        InterEnergyType.DISP,
        InterEnergyType.EX_DISP_OS,
        InterEnergyType.EX_DISP_SS,
        InterEnergyType.DELTA_HF,
    ]

    energy_target_names = [
        "cc_MP2_all",
        "qz_MP2_all",
        "tz_MP2_all",
        "cbs_MP2_all",
        "cc_CCSD(T)_all",
        "cbs_CCSD(T)_all",
        "nn_CCSD(T)_all",
        "sapt_all",
        "sapt_es",
        "sapt_ex",
        "sapt_exs2",
        "sapt_ind",
        "sapt_exind",
        "sapt_disp",
        "sapt_exdisp_os",
        "sapt_exdisp_ss",
        "sapt_delta_HF",
    ]

    @property
    def csv_path(self):
        return os.path.join(self.root, CSV_NAME[self.__name__])

    def read_raw_entries(self) -> List[Dict]:
        filepath = self.csv_path
        logger.info(f"Reading DESS66 interaction data from {filepath}")
        df = pd.read_csv(filepath)
        data = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            item = parse_des_df(row)
            item["subset"] = row["system_name"]
            data.append(convert_to_record(item))
        return data


class DESS66x8(DESS66):
    """
    DE Shaw Research interaction energy
    estimates of all 528 conformers from
    the original S66x8 dataset as described
    in the paper:

    Quantum chemical benchmark databases of gold-standard dimer interaction energies.
    Donchev, A.G., Taube, A.G., Decolvenaere, E. et al.
    Sci Data 8, 55 (2021).
    https://doi.org/10.1038/s41597-021-00833-x

    Data was downloaded from Zenodo:

    https://zenodo.org/records/5676284
    """

    __name__ = "des_s66x8"

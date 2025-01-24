from .alchemy import Alchemy
from .ani import ANI1, ANI1CCX, ANI1CCX_V2, ANI1X, ANI2X
from .bpa import BPA
from .comp6 import COMP6
from .dipeptides import Dipeptides
from .dummy import Dummy, PredefinedDataset
from .gdml import GDML
from .geom import GEOM
from .iso_17 import ISO17
from .maceoff import MACEOFF
from .md22 import MD22
from .molecule3d import Molecule3D
from .multixcqm9 import MultixcQM9, MultixcQM9_V2
from .nabladft import NablaDFT
from .orbnet_denali import OrbnetDenali
from .pcqm import PCQM_B3LYP, PCQM_PM6
from .proteinfragments import MDDataset, ProteinFragments
from .qm1b import QM1B, QM1B_SMALL
from .qm7x import QM7X, QM7X_V2
from .qmugs import QMugs, QMugs_V2
from .qmx import QM7, QM8, QM9, QM7b
from .revmd17 import RevMD17
from .sn2_rxn import SN2RXN
from .solvated_peptides import SolvatedPeptides
from .spice import Spice, SpiceV2, SpiceVL2
from .tmqm import TMQM
from .transition1x import Transition1X
from .vqm24 import VQM24
from .waterclusters import SCANWaterClusters
from .waterclusters3_30 import WaterClusters

AVAILABLE_POTENTIAL_DATASETS = {
    "Alchemy": Alchemy,
    "ANI1": ANI1,
    "ANI1CCX": ANI1CCX,
    "ANI1CCX_V2": ANI1CCX_V2,
    "ANI1X": ANI1X,
    "ANI2X": ANI2X,
    "BPA": BPA,
    "COMP6": COMP6,
    "Dipeptides": Dipeptides,
    "GDML": GDML,
    "GEOM": GEOM,
    "ISO17": ISO17,
    "Molecule3D": Molecule3D,
    "MACEOFF": MACEOFF,
    "NablaDFT": NablaDFT,
    "OrbnetDenali": OrbnetDenali,
    "PCQM_B3LYP": PCQM_B3LYP,
    "PCQM_PM6": PCQM_PM6,
    "QM7X": QM7X,
    "QM7X_V2": QM7X_V2,
    "QMugs": QMugs,
    "QMugs_V2": QMugs_V2,
    "QM1B": QM1B,
    "QM1B_SMALL": QM1B_SMALL,
    "QM7": QM7,
    "QM7b": QM7b,
    "QM8": QM8,
    "QM9": QM9,
    "SN2RXN": SN2RXN,
    "SolvatedPeptides": SolvatedPeptides,
    "Spice": Spice,
    "SpiceV2": SpiceV2,
    "SpiceVL2": SpiceVL2,
    "TMQM": TMQM,
    "Transition1X": Transition1X,
    "WaterClusters": WaterClusters,
    "SCANWaterClusters": SCANWaterClusters,
    "MultixcQM9": MultixcQM9,
    "MultixcQM9_V2": MultixcQM9_V2,
    "RevMD17": RevMD17,
    "MD22": MD22,
    "VQM24": VQM24,
    "ProteinFragments": ProteinFragments,
    "MDDataset": MDDataset,
}

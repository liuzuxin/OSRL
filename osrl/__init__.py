__version__ = "0.1.0"


from osrl.cdt import CDT, CDTTrainer
from osrl.cpq import CPQ, CPQTrainer
from osrl.bcql import BCQL, BCQLTrainer
from osrl.bearl import BEARL, BEARLTrainer
from osrl.coptidice import COptiDICE, COptiDICETrainer


__all__ = [
    "CDT", "CDTTrainer",
    "CPQ", "CPQTrainer",
    "BCQL", "BCQLTrainer",
    "BEARL", "BEARLTrainer",
    "COptiDICE", "COptiDICETrainer"
]
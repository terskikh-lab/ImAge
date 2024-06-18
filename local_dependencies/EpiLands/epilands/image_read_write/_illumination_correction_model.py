from basicpy import BaSiC
from ..generic_read_write import ezsave
from ..generic_read_write import ezload


def save_illumination_correction_model(correction_model: BaSiC, file: str) -> None:
    ezsave(
        {"model": correction_model, "dummy": []},
        file,
    )


def load_illumination_correction_model(file: str) -> BaSiC:
    return ezload(file)["model"]

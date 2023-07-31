import numpy as np

def paramsDict() -> {}:
    """
    abbreviation table = {
        "sdz"    : "SAPT0-D4/cc-pVDZ",
        "sjdz"   : "SAPT0-D4/jun-cc-pVDZ",
        "sadz"   : "SAPT0-D4/aug-cc-pVDZ",
        "stz"    : "SAPT0-D4/cc-pVTZ",
        "satz"   : "SAPT0-D4/aug-cc-pVTZ",

        "sdmjdz" : "SAPT0-D3M(BJ)/jun-cc-pVDZ",
        "sddz"   : "SAPT0-D3(BJ)/cc-pVDZ",
        "sdjdz"  : "SAPT0-D3(BJ)/jun-cc-pVDZ",
        "sdadz"  : "SAPT0-D3(BJ)/aug-cc-pVDZ",
        "sdtz"   : "SAPT0-D3(BJ)/cc-pVTZ",
        "sdatz"  : "SAPT0-D3(BJ)/aug-cc-pVTZ",
    }
    """
    params_dict = {
        "HF": [1.0, 1.61679827, 0.44959224, 3.35743605],
        "HF_ATM": [1.0, 1.61679827, 0.44959224, 3.35743605, 1.0],
        "undamped": [1.0, 1.0, 0.0, 0.0],
        "sddz": [1.0, 0.768328, 0.096991, 3.640770],
        "sdz": [1.0, 0.819671, 0.629414, 1.433599],
        "sdjdz": [1.0, 0.713108, 0.079643, 3.627271],
        "sjdz": [1.0, 0.810710, 0.749569, 0.864432],
        "sdadz": [1.0, 0.732484, 0.094481, 3.632253],
        "sadz": [1.0, 0.829861, 0.706055, 1.123903],
        "sdtz": [1.0, 0.785825, 0.116699, 3.643508],
        "stz": [1.0, 0.846624, 0.629833, 1.525463],
        "sdatz": [1.0, 0.764028, 0.112638, 3.639607],
        "satz": [1.0, 0.844703, 0.668897, 1.350892],
        "sdft_pbe0_adz": [
            1.0,
            1.1223852449709826,
            1.1956254519254155,
            -1.177877609414902,
        ],
        "pbe": [1.0000, 3.64405246, 0.52905620, 4.11311891],
    }
    return params_dict


def get_params(
    params_type="HF",
) -> []:
    """
    get_params gives BJ damping parameters.
    HF parameters come from Grimme's D4 HF parameters in DFTD4 code.

    undamped makes the BJ damping equal 1

    The table below describes what method/basis set the following
    parameters are optimized for with the 8299 dimer dataset (Jeff's).

    table = {
        "sdz"    : "SAPT0-D4/cc-pVDZ",
        "sjdz"   : "SAPT0-D4/jun-cc-pVDZ",
        "sadz"   : "SAPT0-D4/aug-cc-pVDZ",
        "stz"    : "SAPT0-D4/cc-pVTZ",
        "satz"   : "SAPT0-D4/aug-cc-pVTZ",

        "sdmjdz" : "SAPT0-D3M(BJ)/jun-cc-pVDZ",
        "sddz"   : "SAPT0-D3(BJ)/cc-pVDZ",
        "sdjdz"  : "SAPT0-D3(BJ)/jun-cc-pVDZ",
        "sdadz"  : "SAPT0-D3(BJ)/aug-cc-pVDZ",
        "sdtz"   : "SAPT0-D3(BJ)/cc-pVTZ",
        "sdatz"  : "SAPT0-D3(BJ)/aug-cc-pVTZ",
    }
    """
    return np.array(paramsDict()[params_type])

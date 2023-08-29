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
        "HF": [1.0, 1.61679827, 0.44959224, 3.35743605, 0.0],
        "HF_OPT": [1.61679827, 0.44959224, 3.35743605],
        "HF_OPT_2B_START": [
            0.8304747365034967,
            0.7062760278861856,
            1.1237322634625035,
        ],
        "HF_ATM": [
            [1.0, 1.61679827, 0.44959224, 3.35743605, 1.0],
            [1.0, 1.61679827, 0.44959224, 3.35743605, 1.0],
        ],
        "sadz_OPT": [0.829861, 0.706055, 1.123903],
        "HF_ATM_OPT_START": np.array(
            [
                1.61679827,
                0.44959224,
                3.35743605,
                0.44959224,
                3.35743605,
            ],
            dtype=np.float64,
        ),
        "HF_ATM_TT_OPT_START": np.array(
            [
                -0.31,
                3.43,
            ],
            dtype=np.float64,
        ),
        "HF_ATM_CHG_OPT_START": np.array(
            [
                0.72859925,
                1.36668932,
            ],
            dtype=np.float64,
        ),
        "HF_ATM_OPT_OUT": np.array(
            [
                0.8304747365034967,
                0.7062760278861856,
                1.1237322634625035,
                1238.78232231208,
                -0.285365034035284,
            ],
            dtype=np.float64,
        ),
        "HF_2B_ATM_OPT_START": np.array(
            [
                0.829861,
                0.706055,
                1.123903,
                0.44959224,
                3.35743605,
            ],
            dtype=np.float64,
        ),
        "HF_2B_ATM_OPT_START_s9": np.array(
            [
                0.829861,
                0.706055,
                1.123903,
                0.44959224,
                3.35743605,
                1.0,
            ],
            dtype=np.float64,
        ),
        "HF_2B_ATM_OPT_START_s9": np.array(
            [
                0.829861,
                0.706055,
                1.123903,
            ],
            dtype=np.float64,
        ),
        "HF_ATM_SHARED": np.array(
            [
                [1.0, 1.36216693, 0.72859925, 1.36668932, 1.0],
                [1.0, 1.36216693, 0.72859925, 1.36668932, 1.0],
            ],
            dtype=np.float64,
        ),
        # "SAPT0_dz_3_IE_2B": [1.0, 0.819787, 0.628121, 1.439842, 0.0],
        # "SAPT0_jdz_3_IE_2B": [1.0, 0.800006, 0.693751, 1.108349, 0.0],
        # "SAPT0_adz_3_IE_2B": [1.0, 0.830552, 0.706286, 1.123797, 0.0],
        "SAPT0_adz_3_IE_ATM": [1.0, 0.830552, 0.706286, 1.123797, 1.0],
        # "SAPT0_mtz_3_IE_2B": [1.0, 0.859326, 0.716229, 1.132059, 0.0],
        # "SAPT0_jtz_3_IE_2B": [1.0, 0.856697, 0.717010, 1.134837, 0.0],
        # "SAPT0_atz_3_IE_2B": [1.0, 0.856166, 0.718010, 1.136656, 0.0],
        # "SAPT0_tz_3_IE_2B": [1.0, 0.867613, 0.717133, 1.144556, 0.0],
        "SAPT0_adz_3_IE_2B_D3": np.array(
            [
                [1.0, 0.73818347, 0.09542862, 3.63663899, 0.0],
                [1.0, 0.73818347, 0.09542862, 3.63663899, 0.0],
            ]
        ),
        "SAPT0_adz_3_IE_2B": np.array(
            [
                [1.0, 0.83055196, 0.70628586, 1.12379695, 0.0],
                [1.0, 0.83055196, 0.70628586, 1.12379695, 0.0],
            ]
        ),
        "SAPT0_jdz_3_IE_2B_D3": np.array(
            [
                [1.0, 0.71425792, 0.07869063, 3.63319367, 0.0],
                [1.0, 0.71425792, 0.07869063, 3.63319367, 0.0],
            ]
        ),
        "SAPT0_jdz_3_IE_2B": np.array(
            [
                [1.0, 0.80000568, 0.69375056, 1.10834867, 0.0],
                [1.0, 0.80000568, 0.69375056, 1.10834867, 0.0],
            ]
        ),
        "SAPT0_mtz_3_IE_2B_D3": np.array(
            [
                [1.0, 0.76414494, 0.10717688, 3.64502602, 0.0],
                [1.0, 0.76414494, 0.10717688, 3.64502602, 0.0],
            ]
        ),
        "SAPT0_mtz_3_IE_2B": np.array(
            [
                [1.0, 0.85932558, 0.71622944, 1.13205874, 0.0],
                [1.0, 0.85932558, 0.71622944, 1.13205874, 0.0],
            ]
        ),
        "SAPT0_jtz_3_IE_2B_D3": np.array(
            [
                [1.0, 0.76220823, 0.10866246, 3.64634887, 0.0],
                [1.0, 0.76220823, 0.10866246, 3.64634887, 0.0],
            ]
        ),
        "SAPT0_jtz_3_IE_2B": np.array(
            [
                [1.0, 0.85669728, 0.71700985, 1.13483727, 0.0],
                [1.0, 0.85669728, 0.71700985, 1.13483727, 0.0],
            ]
        ),
        "SAPT0_dz_3_IE_2B_D3": np.array(
            [
                [1.0, 0.76805484, 0.0969723, 3.64297355, 0.0],
                [1.0, 0.76805484, 0.0969723, 3.64297355, 0.0],
            ]
        ),
        "SAPT0_dz_3_IE_2B": np.array(
            [
                [1.0, 0.81978749, 0.62812067, 1.43984245, 0.0],
                [1.0, 0.81978749, 0.62812067, 1.43984245, 0.0],
            ]
        ),
        "SAPT0_atz_3_IE_2B_D3": np.array(
            [
                [1.0, 0.76344431, 0.11057267, 3.64771308, 0.0],
                [1.0, 0.76344431, 0.11057267, 3.64771308, 0.0],
            ]
        ),
        "SAPT0_atz_3_IE_2B": np.array(
            [
                [1.0, 0.85616613, 0.71801039, 1.13665562, 0.0],
                [1.0, 0.85616613, 0.71801039, 1.13665562, 0.0],
            ]
        ),
        "SAPT0_tz_3_IE_2B_D3": np.array(
            [
                [1.0, 0.78563464, 0.11449826, 3.65265882, 0.0],
                [1.0, 0.78563464, 0.11449826, 3.65265882, 0.0],
            ]
        ),
        "SAPT0_tz_3_IE_2B": np.array(
            [
                [1.0, 0.86761307, 0.71713316, 1.14455592, 0.0],
                [1.0, 0.86761307, 0.71713316, 1.14455592, 0.0],
            ]
        ),
        "undamped": [1.0, 1.0, 0.0, 0.0, 0.0],
        "sddz": [1.0, 0.768328, 0.096991, 3.640770, 0.0],
        "sdz": [1.0, 0.819671, 0.629414, 1.433599, 0.0],
        "sdjdz": [1.0, 0.713108, 0.079643, 3.627271, 0.0],
        "sjdz": [1.0, 0.810710, 0.749569, 0.864432, 0.0],
        "sdadz": [1.0, 0.732484, 0.094481, 3.632253, 0.0],
        "sadz": [1.0, 0.829861, 0.706055, 1.123903, 0.0],
        "sdtz": [1.0, 0.785825, 0.116699, 3.643508, 0.0],
        "stz": [1.0, 0.846624, 0.629833, 1.525463, 0.0],
        "sdatz": [1.0, 0.764028, 0.112638, 3.639607, 0.0],
        "satz": [1.0, 0.844703, 0.668897, 1.350892, 0.0],
        "sdft_pbe0_adz": [
            1.0,
            1.1223852449709826,
            1.1956254519254155,
            -1.177877609414902,
            0.0,
        ],
        "pbe": [1.0000, 3.64405246, 0.52905620, 4.11311891, 0.0],
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
    return np.array(paramsDict()[params_type], dtype=np.float64)


def generate_2B_ATM_param_subsets(
    params,
    params_2B_key="HF_ATM_SHARED",  # "SAPT0_adz_3_IE_2B",
    force_ATM_on=False,
):
    """
    params types:
        7 params = [s6, s8, a1, a2, a1_ATM, a2_ATM, s9],
        6 params = [s8, a1, a2, a1_ATM, a2_ATM, s9],
        5 params = [s6, s8, a1, a2, s9] || [s8, a1, a2, a1_ATM, a2_ATM],
        4 params = [s6, s8, a1, a2],
        2 params = [
                [s6, s8, a1, a2],
                [s6, s8, a1_ATM, a2_ATM],
        ]
    """
    s9 = 0.0
    if force_ATM_on:
        s9 = 1.0
    if len(params) == 7:
        params_2B = np.array(params[:4], dtype=np.float64)
        params_ATM = np.array(
            [params[0], params[1], params[4], params[5], params[6]], dtype=np.float64
        )
    elif len(params) == 6:
        params_2B = np.array(
            [1.0, params[0], params[1], params[2], params[5]], dtype=np.float64
        )
        params_ATM = np.array(
            [1.0, params[0], params[3], params[4], params[5]], dtype=np.float64
        )
    elif (
        len(params) == 5
        and abs(params[-1] - 1.0) < 1e-6
        and abs(params[0] - 1.0) < 1e-6
    ):
        # print("Special 5 ATM")
        params_2B = np.array(params, dtype=np.float64)
        params_ATM = np.array(params, dtype=np.float64)
    elif len(params) == 5 and abs(params[-1]) < 1e-6 and abs(params[0] - 1.0) < 1e-6:
        # print("Special 5 2body")
        params_2B = np.array(params, dtype=np.float64)
        params_ATM = np.array(params, dtype=np.float64)
    elif len(params) == 5:
        params_2B = np.array(
            [1.0, params[0], params[1], params[2], s9], dtype=np.float64
        )
        params_ATM = np.array(
            [1.0, params[0], params[3], params[4], s9], dtype=np.float64
        )
    elif len(params) == 3:

        params_2B = np.array(
            [1.0, params[0], params[1], params[2], s9], dtype=np.float64
        )
        params_ATM = np.array(
            [1.0, params[0], params[1], params[2], s9], dtype=np.float64
        )
    elif len(params) == 2 and (
        type(params[0]) == list or type(params[0]) == np.ndarray
    ):
        params_2B = np.array(params[0], dtype=np.float64)
        params_ATM = np.array(params[1], dtype=np.float64)
    elif len(params) == 2:
        params_2B, params_ATM = get_params(params_2B_key)
        params_ATM[2] = params[0]
        params_ATM[3] = params[1]
        print(params_2B, params_ATM)
    else:
        print(len(params), params)
        raise ValueError("params must be of size 2, 3, 5, 6, or 7!")
    return params_2B, params_ATM

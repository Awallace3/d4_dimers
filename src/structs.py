from dataclasses import dataclass
from dataclasses import dataclass
import numpy as np

@dataclass
class d4_values:
    C6s: [np.array]
    Qs: [np.array]
    C8s: [np.array]

@dataclass
class method_info:
    method: str
    monA: float
    monB: float
    dimer: float
    int_e: float
    params: []


@dataclass
class dimers:
    System: str
    z: str
    DB: str
    C6s: np.array
    C8s: np.array
    Benchmark: float
    HF_jun_DZ: float
    HF_aug_DZ: float
    HF_cc_PVDZ: float
    HF_cc_PTDZ: float
    HF_aug_cc_PTDZ: float


@dataclass
class mol_i:
    energies: np.array
    atom_orders: np.array
    carts: np.array
    C6s: np.array
    Qs: np.array
    C8s: np.array

    def print_cartesians(self):
        """
        prints a 2-D numpy array in a nicer format
        """
        arr = np.zeros((len(self.carts), 4))
        arr[:, 0] = self.atom_orders
        arr[:, 1:] = self.carts
        for a in arr:
            for i, elem in enumerate(a):
                if i == 0:
                    print("{} ".format(int(elem)), end="")
                else:
                    print("{} ".format(elem).rjust(3), end="")
            print(end="\n")


@dataclass
class mols_data:
    energies: np.array
    atom_orders: [np.array]
    carts: [np.array]
    d4_vals: d4_values

    def get_mol(self, pos):
        return mol_i(
            self.energies[pos],
            self.atom_orders[pos],
            self.carts[pos],
            self.d4_vals.C6s[pos],
            self.d4_vals.Qs[pos],
            self.d4_vals.C6s[pos],
        )


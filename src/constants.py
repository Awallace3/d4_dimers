import math


class Constants:
    def __init__(self):
        electron_mass = 9.1093837015e-31
        c = 299792458
        fine_structure_constant = 7.2973525693e-3
        hbar = 6.62607015e-34 / (2 * math.pi)
        bohr = hbar / (electron_mass * c * fine_structure_constant)
        autoaa = bohr * 1e10
        aatoau = 1 / autoaa

        self.em = electron_mass
        self.c = c
        self.fsc = fine_structure_constant
        self.hbar = hbar
        self.bohr = bohr
        self.autoaa = autoaa
        self.aatoau = aatoau

    def g_em(self):
        return self.em

    def g_c(self):
        return self.c

    def g_fsc(self):
        return self.fsc

    def g_hbar(self):
        return self.hbar

    def g_bohr(self):
        return self.bohr

    def g_autoaa(self):
        return self.autoaa

    def g_aatoau(self):
        return self.aatoau

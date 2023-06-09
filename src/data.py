import qcelemental as qcel
import numpy as np

def water_geom():
    coords = np.array(
        [
            [8, 0.0370878820, 0.0000000000, -0.0551122700],
            [1, -0.7006692913, 0.5931409324, 0.1638729949],
            [1, 0.1120576564, -0.5931409324, 0.7107987731],
            [8, -0.0116916226, -0.0550639593, 2.2876732163],
            [1, 0.7884890539, 0.1683899814, 2.7914941964],
            [1, -0.6029345083, 0.7055150610, 2.4141950038],
        ]
    )
    num = coords[:, 0]
    ang_to_bohr = qcel.constants.conversion_factor("angstrom", "bohr")
    coords = coords[:, 1:] # * ang_to_bohr
    return num, coords

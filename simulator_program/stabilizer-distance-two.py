# Code intended to reproduce the experiment/simulation done by Delft (Logical-qubit operations in an error-detecting surface code)
# and/or (Repeated quantum error detection in a surface code) on the distance 2 stabilizer code with post selection.

# %% Import modules
import numpy as np
from qiskit import (QuantumCircuit,
                    QuantumRegister,
                    ClassicalRegister,
                    AncillaRegister,
                    execute,
                    Aer
                    )
from qiskit.providers.aer.extensions.snapshot_statevector import *
from qiskit.providers.aer.extensions.snapshot_density_matrix import *
from custom_transpiler import *
from typing import List
# %% Logical states (for initialization)


def logical_states(include_ancillas='front') -> List[List[float]]:
    """Returns the logical states for the distance two code.

    Args:
        include_ancillas (str/None, optional): Whether to append the ancillas by tensor product to the end. Defaults to True.

    Returns:
        List[List[float]]: List of both logical states
    """
    logical_0 = np.zeros(2**4)
    logical_0[0b0000] = 1/np.sqrt(2)
    logical_0[0b1111] = 1/np.sqrt(2)

    logical_1 = np.zeros(2**4)
    logical_1[0b0101] = 1/np.sqrt(2)
    logical_1[0b1010] = 1/np.sqrt(2)

    if include_ancillas:
        # Add two ancillas in |0>
        an0 = np.zeros(2**2)
        an0[0] = 1.0
        if include_ancillas == 'front':
            logical_1 = np.kron(logical_1, an0)
            logical_0 = np.kron(logical_0, an0)
        elif include_ancillas == 'back':
            logical_1 = np.kron(an0, logical_1)
            logical_0 = np.kron(an0, logical_0)

    return [logical_0, logical_1]

# %% Custom circuits

class StabilizerRegisters:
    def __init__(self,
                 qbReg=QuantumRegister(4, 'code_qubit'),
                 anReg=AncillaRegister(3, 'ancilla_qubit'),
                 clReg=ClassicalRegister(3, 'syndrome_bit'),
                 readout=ClassicalRegister(5, 'readout')
                 ):
        self.QubitRegister = qbReg
        self.AncillaRegister = anReg
        self.SyndromeRegister = clReg
        self.ReadoutRegister = readout

def pipelined_delft(registers, n_cycles=1, **kwargs):
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    readout = registers.ReadoutRegister

    circ = QuantumCircuit(qbReg, anReg, clReg, readout)
    circ.set_statevector(logical_states()[0])

    for cycle in range(n_cycles):
        # Blue half, (XXXX) on the D register
        # TODO: add 'parking'?
        for D in qbReg:
            circ.ry(-1*np.pi/2, D)
        circ.ry(-1*np.pi/2, anReg[1])

        circ.cz(qbReg[1], anReg[1])
        circ.cz(qbReg[0], anReg[1])
        circ.cz(qbReg[3], anReg[1])
        circ.cz(qbReg[2], anReg[1])

        for D in qbReg:
            circ.ry(np.pi/2, D)
        circ.ry(np.pi/2, anReg[1])
        circ.measure(anReg[1],clReg[0]) # TODO: fix

        # Blue half, (XXXX) on the D register
# %% Custom noise models

# Info of T1/T2 and gate times is the Mendeley paper
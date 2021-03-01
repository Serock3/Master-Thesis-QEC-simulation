'''Comparing the performance of using a large number of shots compared to
iterating outside of the simulation.'''

# %% Import modules
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import pauli_error,depolarizing_error
from qiskit.providers.aer.noise import (thermal_relaxation_error, 
                                        amplitude_damping_error, 
                                        phase_damping_error, 
                                        phase_amplitude_damping_error
                                        )

from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Our own files
from simulator_program.custom_noise_models import phase_amplitude_model
from simulator_program.stabilizers import *


# %% =================  Testing noise model + stabilizer

def many_shots():
    # Define our registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = ClassicalRegister(5, 'syndrome_bit') # The typical register
    readout = ClassicalRegister(5, 'readout')

    registers = StabilizerRegisters(qb, an, cr, readout)
    circ = get_empty_stabilizer_circuit(registers)

    # Settings for circuit
    n_cycles = 1
    reset=False
    flag=False
    recovery=False

    # Get the circuit
    circ.x(qb[0])
    circ += get_full_stabilizer_circuit(registers,
        n_cycles=n_cycles,
        reset=reset,
        recovery=recovery,
        flag=flag,
    )
    # Analyze results
    n_shots = 2048*32

    # Run the circuit
    results = execute(
        circ,  
        Aer.get_backend('qasm_simulator'),
        noise_model=phase_amplitude_model(),
        shots=n_shots
    ).result()
    counts = results.get_counts()

def single_shot():
    # Define our registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = ClassicalRegister(5, 'syndrome_bit') # The typical register
    readout = ClassicalRegister(5, 'readout')

    registers = StabilizerRegisters(qb, an, cr, readout)
    circ = get_empty_stabilizer_circuit(registers)

    # Settings for circuit
    n_cycles = 1
    reset=False
    flag=False
    recovery=False

    # Get the circuit
    circ.x(qb[0])
    circ += get_full_stabilizer_circuit(registers,
        n_cycles=n_cycles,
        reset=reset,
        recovery=recovery,
        flag=flag,
    )
    # Analyze results
    n_shots = 2048*32
    for i in range(n_shots):
        # Run the circuit
        results = execute(
            circ,  
            Aer.get_backend('qasm_simulator'),
            noise_model=phase_amplitude_model(),
            shots=1
        ).result()
        counts = results.get_counts()


# %%
def test():
    """Stupid test function"""
    lst = []
    for i in range(100):
        lst.append(i)

if __name__ == '__main__':
    import timeit
    print('Time for single shot:')
    print(timeit.timeit("single_shot()", setup="from __main__ import single_shot", number=1))

    print('Time for multi shot:')
    print(timeit.timeit("single_shot()", setup="from __main__ import single_shot", number=1))
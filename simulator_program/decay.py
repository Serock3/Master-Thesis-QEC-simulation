""" This file creates circuits for simulating decays for encoded and single qubits.
Functions are mainly used in active_qec.ipynb and 422_code_summary.ipynb
"""
#%% Imports
import numpy as np
from simulator_program.data_analysis_tools import fidelity_from_scratch
from matplotlib import pyplot as plt
from qiskit.providers.aer.noise.errors.standard_errors import thermal_relaxation_error
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.execute_function import execute
from qiskit.providers.aer.library import save_density_matrix
from qiskit import Aer
from simulator_program.stabilizers import add_snapshot_to_circuit, logical_states
from qiskit.quantum_info.states.measures import state_fidelity
import scipy
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator
from qiskit.providers.aer.noise import kraus_error

from .stabilizers_422 import get_encoded_state_422
#%% 
def get_idle_single_qubit(snapshot_times, snapshot_type='dm', T1=40e3, T2=60e3,
                          theta=0, phi=0, pauliop='Z'):
    """Generates and runs a single qubit-circuit initialized in the specified state with
    snapshots at given times. 

    Returns:
        result: Qiskit results object.
    """
    qb = QuantumRegister(1, 'qubit')
    circ = QuantumCircuit(qb)
    initial_state = np.cos(theta/2)*np.array((1,0)) + \
        np.exp(1j*phi)*np.sin(theta/2)*np.array((0,1))
    circ.set_density_matrix(initial_state)
    time_passed = 0
    for i, time in enumerate(snapshot_times):
        time_diff = time-time_passed
        if time_diff > 0:
            thrm_relax = thermal_relaxation_error(
                T1, T2, time_diff).to_instruction()

            circ.append(thrm_relax, [qb[0]])
        add_snapshot_to_circuit(circ, snapshot_type, i, [
                                qb[0]], conditional=False, pauliop=pauliop)
        time_passed = time

    simulator = Aer.get_backend('aer_simulator')
    simulator.set_option('method', 'density_matrix')
    results = execute(circ, simulator,
                      noise_model=None, shots=1).result()
    return results


def get_idle_encoded_513(snapshot_times, snapshot_type='dm', T1=40e3, T2=60e3,
                         theta=0, phi=0, pauliop='ZZZZZ'):
    """Initializes a logical qubit in the desired state and lets it decay under
    thermal relaxation noise. Takes snapshots of the density matrix (or an
    expectation value) at specified times.
    """
    
    logical_0, logical_1 = logical_states(include_ancillas=None)
    circ = QuantumCircuit(5)
    initial_state = np.cos(theta/2)*logical_0 + \
        np.exp(1j*phi)*np.sin(theta/2)*logical_1
    circ.set_density_matrix(initial_state)
    time_passed = 0
    for i, time in enumerate(snapshot_times):
        time_diff = time-time_passed
        if time_diff > 0:
            thrm_relax = thermal_relaxation_error(
                T1, T2, time_diff).to_instruction()
            for qubit in circ.qubits:
                circ.append(thrm_relax, [qubit])

        add_snapshot_to_circuit(circ, snapshot_type, i,
                                circ.qubits, conditional=False, pauliop=pauliop)
        time_passed = time

    simulator = Aer.get_backend('aer_simulator')
    simulator.set_option('method', 'density_matrix')
    results = execute(circ, simulator,
                      noise_model=None, shots=1).result()
    return results

def get_idle_encoded_422(snapshot_times, snapshot_type='dm', T1=40e3, T2=60e3,
                         initial_state=[1., 0., 0., 0.], pauliop='ZZZZZ'):
    """Initializes a logical qubit in the desired state and lets it decay under
    thermal relaxation noise. Takes snapshots of the density matrix (or an
    expectation value) at specified times.
    """

    circ = QuantumCircuit(4)
    initial_state = get_encoded_state_422(initial_state, include_ancillas=None)
    circ.set_density_matrix(initial_state)
    time_passed = 0
    for i, time in enumerate(snapshot_times):
        time_diff = time-time_passed
        if time_diff > 0:
            thrm_relax = thermal_relaxation_error(
                T1, T2, time_diff).to_instruction()
            for qubit in circ.qubits:
                circ.append(thrm_relax, [qubit])

        add_snapshot_to_circuit(circ, snapshot_type, i,
                                circ.qubits, conditional=False, pauliop=pauliop)
        time_passed = time

    simulator = Aer.get_backend('aer_simulator')
    simulator.set_option('method', 'density_matrix')
    results = execute(circ, simulator,
                      noise_model=None, shots=1).result()
    return results
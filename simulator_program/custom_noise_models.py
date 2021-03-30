# Import from Qiskit Aer noise module 
from qiskit.providers.aer.noise import (
    pauli_error,
    depolarizing_error,
    thermal_relaxation_error, 
    amplitude_damping_error, 
    phase_damping_error , 
    phase_amplitude_damping_error,
)
from qiskit.providers.aer.noise import NoiseModel, QuantumError, ReadoutError
from qiskit.providers.aer import noise
import numpy as np
from qiskit.circuit.library import standard_gates
from inspect import getmembers, isfunction, isclass, ismodule

def get_standard_gate_times():
    """Return a dict of standard gate times (ns) used for simulator purposes."""
    return {
        'x': 20, 'y': 20, 'z': 0, 'h': 20, 'u1': 0, 'u2': 20, 'u3': 20,
        'cx': 200, 'cz': 200, 'swap': 200, 'iswap': 200,
        'barrier': 0, 'measure': 500, 'snapshot': 0
    }

## Define noise models

def pauli_noise_model(p_gate1=0.0, p_meas=0.0, p_reset=0.0):
    '''Testing around with some Bit-flip noise'''

    # QuantumError objects
    error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    error_gate1 = pauli_error([('X', p_gate1), ('I', 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model
    noise_bit_flip = NoiseModel()
    noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
    noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
    noise_bit_flip.add_all_qubit_quantum_error(error_gate1,
                                               ["u1", "u2", "u3", "x", "z", "h"])
    noise_bit_flip.add_all_qubit_quantum_error(
        error_gate2, ["cx", "cz", "swap", "iswap"])

    return noise_bit_flip

def thermal_relaxation_model(T1=40e3, T2=60e3, t_single=15, t_cz=300,
        t_measure=1000, t_reset=1000):
    """Noise model for thermal relaxation. All times are given
    in nanoseconds (ns).
    """
    # TODO: Add accurate times for reset/measurement. Current ones use 
    #       example from qiskit


    # QuantumError objects
    error_single = thermal_relaxation_error(T1, T2, t_single)
    error_measure = thermal_relaxation_error(T1, T2, t_measure)
    error_reset = thermal_relaxation_error(T1, T2, t_reset)
    error_cz = thermal_relaxation_error(T1, T2, t_cz).expand(
        thermal_relaxation_error(T1, T2, t_cz))

    # Add errors to noise model
    noise_damping = NoiseModel()
    #noise_damping.add_all_qubit_quantum_error(error_measure, "measure")
    #noise_damping.add_all_qubit_quantum_error(error_reset, "reset")
    noise_damping.add_all_qubit_quantum_error(error_single,
        ["x", "z", "h", "id", "u1", "u2","u3"])
    noise_damping.add_all_qubit_quantum_error(error_cz, 
        ["cx", "cz", "swap", "iswap"])

    return noise_damping


def phase_amplitude_model(T1=40e3, T2=60e3, t_single=15, t_cz=300,
        t_measure=1000, t_reset=1000):
    """Noise model for amplitude and phase damping. All times are given
    in nanoseconds (ns).
    """
    # TODO: Add accurate times for reset/measurement. Current ones use 
    #       example from qiskit

    # Calculate parameters for both 1-qb and 2-qb gates
    pAD_single = 1 - np.exp(-t_single/T1)
    pPD_single = 1 - np.exp(-2*t_single/T2) / np.exp(-t_single/T1)

    pAD_cz = 1-np.exp(-t_cz/T1)
    pPD_cz = 1 - (np.exp(-2*t_cz/T2))/(np.exp(-t_cz/T1))

    pAD_measure = 1 - np.exp(-t_measure/T1)
    pPD_measure = 1 - np.exp(-2*t_measure/T2) / np.exp(-t_measure/T1)

    pAD_reset = 1 - np.exp(-t_reset/T1)
    pPD_reset = 1 - np.exp(-2*t_reset/T2) / np.exp(-t_reset/T1)

    # QuantumError objects
    error_single = phase_amplitude_damping_error(pAD_single, pPD_single)
    error_measure = phase_amplitude_damping_error(pAD_measure, pPD_measure)
    error_reset = phase_amplitude_damping_error(pAD_single, pPD_reset)
    error_cz = phase_amplitude_damping_error(pAD_cz, pPD_cz).expand(
        phase_amplitude_damping_error(pAD_cz, pPD_cz))

    # Add errors to noise model
    noise_damping = NoiseModel()
    noise_damping.add_all_qubit_quantum_error(error_measure, "measure")
    noise_damping.add_all_qubit_quantum_error(error_reset, "reset")
    noise_damping.add_all_qubit_quantum_error(error_single,
        ["x", "z", "h", "id", "u1", "u2","u3"])
    noise_damping.add_all_qubit_quantum_error(error_cz, 
        ["cx", "cz", "swap", "iswap"])

    return noise_damping

# Check this link for reference
# https://qiskit.org/documentation/tutorials/simulators/3_building_noise_models.html

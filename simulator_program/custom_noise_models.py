# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import pauli_error,depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error, amplitude_damping_error, phase_damping_error , phase_amplitude_damping_error
from qiskit.providers.aer.noise import NoiseModel,QuantumError, ReadoutError
from qiskit.providers.aer import noise
import numpy as np


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


def thermal_relaxation_model():
    # T1 and T2 values for qubits 0-3
    # Sampled from normal distribution mean 50 microsec
    T1s = np.random.normal(50e3, 10e3, 4)
    # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal(70e3, 10e3, 4)

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(4)])

    # Instruction times (in nanoseconds)
    time_u1 = 0   # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100  # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000  # 1 microsecond

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                      for t1, t2 in zip(T1s, T2s)]
    errors_u1 = [thermal_relaxation_error(t1, t2, time_u1)
                 for t1, t2 in zip(T1s, T2s)]
    errors_u2 = [thermal_relaxation_error(t1, t2, time_u2)
                 for t1, t2 in zip(T1s, T2s)]
    errors_u3 = [thermal_relaxation_error(t1, t2, time_u3)
                 for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
        thermal_relaxation_error(t1b, t2b, time_cx))
        for t1a, t2a in zip(T1s, T2s)]
        for t1b, t2b in zip(T1s, T2s)]

    # Add errors to noise model
    noise_thermal = NoiseModel()

    for j in range(4):
        noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
        noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
        noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
        noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
        noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
        for k in range(4):
            noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])
    return noise_thermal

# def phase_amplitude_damping_model():


#     errors_u1 = [phase_amplitude_damping_error(t1, t2, time_u1)
#                  for t1, t2 in zip(T1s, T2s)]
# Check this link for reference https://qiskit.org/documentation/tutorials/simulators/3_building_noise_models.html

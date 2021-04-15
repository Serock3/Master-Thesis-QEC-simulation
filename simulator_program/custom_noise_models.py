# %%
# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import (
    pauli_error,
    depolarizing_error,
    thermal_relaxation_error,
    amplitude_damping_error,
    phase_damping_error,
    phase_amplitude_damping_error,
)
from qiskit.providers.aer.noise import NoiseModel, QuantumError, ReadoutError
from qiskit.providers.aer import noise
import numpy as np
from qiskit.circuit.library import standard_gates
from qiskit.circuit import Gate
import warnings

# %%


class GateTimes:
    """Class to contain gate times.
        Auto populates the gate_times dictionary with (specified) default values for single and two qubit gates.
    """
    # Lists of gate names included in noise models and their type.
    # Gates not in these sets will not be possible to include in noise models.
    # This will help noise models know how many qubits to apply noise to.
    # NOTE: Can this be generated automatically from standard_gates?
    single_qubit_gates = {'x', 'y', 'z', 'h', 'sx',
                          'sz', 'sy', 's', 't', 'u1', 'u2', 'u3'}
    two_qubit_gates = {'cx', 'cz', 'swap', 'iswap'}
    special_ops = {'measure', 'reset'}
    directives = {'barrier', 'set_density_matrix',
                  'save_density_matrix', 'save_expval', 'snapshot'}

    def __init__(self, single_qubit_default=0, two_qubit_default=0, custom_gate_times={}):
        """Class to contain a dictionary of gate times. 
        Adds the gate times specified in custom_gate_times and 
        auto populates the rest with specified default values 
        for single and two qubit gates.

        Args:
            single_qubit_default (int, optional): Gate time for any single qubit gate not specified in custom_gate_times. Defaults to 0.
            two_qubit_default (int, optional): Gate time for any two qubit gate not specified in custom_gate_times. Defaults to 0.
            custom_gate_times (dict, optional): Specify any desired gate times. Defaults to {}.
        """
        self.gate_times = {
            name: single_qubit_default for name in GateTimes.single_qubit_gates}
        self.gate_times.update(
            {name: two_qubit_default for name in GateTimes.two_qubit_gates})
        self.gate_times.update({name: 0 for name in GateTimes.special_ops})
        self.gate_times.update({name: 0 for name in GateTimes.directives})
        self.gate_times.update({name.lower(): time for (
            name, time) in custom_gate_times.items()})

    def __getitem__(self, item):
        return self.gate_times[item]

    def get_gate_times(self, custom_gate_times={}):
        """Return a dict of standard gate (and operator) times (ns). Use to add time dependent noise.

        Args:
            custom_gate_times (dict, optional): Overwrite any preset gate times with these. Defaults to {}.

        Returns:
            dict: gate times in ns for the name str of each gate and op.
        """

        gate_times_tmp = self.gate_times.copy()
        gate_times_tmp.update(custom_gate_times)
        return gate_times_tmp

    def get_single_qubit_times(self):
        return {name: self.gate_times[name] for name in GateTimes.single_qubit_gates}

    def get_two_qubit_times(self):
        return {name: self.gate_times[name] for name in GateTimes.two_qubit_gates}

    def __repr__(self):
        return f"GateTimes(0, 0, {self.gate_times.__repr__()})"

    def __str__(self):
        return "GateTimes object with times (ns)\n" + self.gate_times.__str__()


# TODO: Here we can define e.g. WACQT_targeted_gate_times, and other versions
WACQT_gate_times = GateTimes(
    single_qubit_default=20, two_qubit_default=200,
    custom_gate_times={'u1': 0, 'z': 0, 'measure': 500})

WACQT_target_times = GateTimes(
    single_qubit_default=20, two_qubit_default=100,
    custom_gate_times={'u1': 0, 'z': 0, 'measure': 300})

WACQT_demonstrated_times = GateTimes(
    single_qubit_default=20, two_qubit_default=300,
    custom_gate_times={'u1': 0, 'z': 0, 'measure': 2300})


# Define noise models

def thermal_relaxation_model_V2(T1=40e3, T2=60e3, gate_times=WACQT_gate_times):
    """Noise model for thermal relaxation. All times are given
    in nanoseconds (ns).

    Args:
        T1 (int, optional): . Defaults to 40e3.
        T2 (int, optional):  Defaults to 60e3.
        gate_times (dict/GateTimes object, optional): Gate times, defaults to standard WACQT_gate_times.

    Returns:
        Noise model: thermal relaxation noise model
    """

    # Convert from dict object to GateTimes object
    if isinstance(gate_times, dict):
        gate_times = GateTimes(
            0, 0, WACQT_gate_times.get_gate_times(gate_times))

    noise_damping = NoiseModel()

    # NOTE: Moved this to class method
    # time_group = {}
    # # Fill with empty lists for each unique time
    # for single_qubit_gate in GateTimes.single_qubit_gates:
    #     time_group[gate_times[single_qubit_gate]] = []

    # for single_qubit_gate in GateTimes.single_qubit_gates:
    #     time_group[gate_times[single_qubit_gate]].append(single_qubit_gate)

    # Add single qubit error in groups of same duration
    # Add errors to noise model

    time_group = {}
    for k, v in gate_times.get_single_qubit_times().items():
        time_group[v] = time_group.get(v, []) + [k]

    for t_single in time_group:
        error_single = thermal_relaxation_error(T1, T2, t_single)
        noise_damping.add_all_qubit_quantum_error(error_single,
                                                  time_group[t_single])

    # Do the same for two qubit gates
    time_group = {}
    for k, v in gate_times.get_two_qubit_times().items():
        time_group[v] = time_group.get(v, []) + [k]

    for t_double in time_group:
        error_double = thermal_relaxation_error(T1, T2, t_double).expand(
            thermal_relaxation_error(T1, T2, t_double))
        noise_damping.add_all_qubit_quantum_error(error_double,
                                                  time_group[t_double])

    # NOTE: More consistent to loop over special_ops?
    error_reset = thermal_relaxation_error(T1, T2, gate_times['reset'])
    noise_damping.add_all_qubit_quantum_error(error_reset, "reset")
    error_measure = thermal_relaxation_error(T1, T2, gate_times['measure'])
    noise_damping.add_all_qubit_quantum_error(error_measure, "measure")

    return noise_damping


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
    warnings.warn(
        "Use thermal_relaxation_model_V2 instead for more accurate gate times",
        DeprecationWarning
    )
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
                                              ["x", "z", "h", "id", "u1", "u2", "u3"])
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
                                              ["x", "z", "h", "id", "u1", "u2", "u3"])
    noise_damping.add_all_qubit_quantum_error(error_cz,
                                              ["cx", "cz", "swap", "iswap"])

    return noise_damping

# Check this link for reference
# https://qiskit.org/documentation/tutorials/simulators/3_building_noise_models.html


# %%
if __name__ == '__main__':
    from qiskit import QuantumCircuit, execute, Aer
    thermal_relaxation_model()

    noise_model = pauli_noise_model(p_gate1=1)

    circ = QuantumCircuit(3)
    circ.x(0)
    circ.swap(0, 1)
    circ.measure_all()
    print(circ)
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=noise_model,
        shots=10
    ).result()
    print(results.get_counts())
# %%

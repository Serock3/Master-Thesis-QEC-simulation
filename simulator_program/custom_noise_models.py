"""File containing functions for building different noise models to be applied
in simulations. For an introduction to noise in Qiskit, see
https://qiskit.org/documentation/tutorials/simulators/3_building_noise_models.html
"""
# %%
# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import (
    pauli_error,
    thermal_relaxation_error,
    phase_amplitude_damping_error
)
from qiskit.providers.aer.noise import NoiseModel
import numpy as np
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
                          'sz', 'sy', 's', 't', 'ry', 'u1', 'u2', 'u3'}
    two_qubit_gates = {'cx', 'cz', 'swap', 'iswap'}
    special_ops = {'measure', 'reset', 'unitary', 'split'}
    directives = {'barrier', 'set_density_matrix', 'save_density_matrix', 
                  'save_expval', 'save_expval_var', 'snapshot', 'feedback',
                  'delay'}

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
            dict[str,int]: gate times in ns for the name str of each gate and op.
        """

        gate_times_tmp = self.gate_times.copy()
        gate_times_tmp.update(custom_gate_times)
        return gate_times_tmp

    def get_single_qubit_times(self):
        """Return a dictionary of single qubit gate times where the key is the gate name 'str',
        and the value is the time in ns.

        Returns:
            dict[str,int]: Gate times (ns).
        """
        return {name: self.gate_times[name] for name in GateTimes.single_qubit_gates}

    def get_two_qubit_times(self):
        """Return a dictionary of two qubit gate times where the key is the gate name 'str',
        and the value is the time in ns.

        Returns:
            dict[str,int]: Gate times (ns).
        """
        return {name: self.gate_times[name] for name in GateTimes.two_qubit_gates}

    def __repr__(self):
        return f"GateTimes(0, 0, {self.gate_times.__repr__()})"

    def __str__(self):
        return "GateTimes object with times (ns)\n" + self.gate_times.__str__()


# Some various setups of different gate times. In practice, Z-gate (or U1) is
# instant as it can be embedded into adjacent gates as an added phase.
WACQT_gate_times = GateTimes(
    single_qubit_default=20, two_qubit_default=200,
    custom_gate_times={'u1': 0, 'z': 0, 'measure': 500})

WACQT_target_times = GateTimes(
    single_qubit_default=20, two_qubit_default=100,
    custom_gate_times={'u1': 0, 'z': 0, 'measure': 300})

standard_times = GateTimes(
    single_qubit_default=20, two_qubit_default=100,
    custom_gate_times={'u1': 0, 'z': 0, 'measure': 300, 'feedback': 350})

standard_times_delay = GateTimes(
    single_qubit_default=20, two_qubit_default=100,
    custom_gate_times={'u1': 0, 'z': 0, 'measure': 300, 'feedback': 350, 'delay': 5000})
# Define noise models

def extend_standard_gate_times(gate_times={}):
    """Adds the specified gate times to a list of standard gate times.
    TODO: Default argument should be None for stability reasons.

    Args:
        gate_times (dict, optional): List of gate (operation) times in ns, e.g. {'cz':200}. Defaults to {}.

    Returns:
        dict: Complete list of times for all gates.
    """
    if isinstance(gate_times, dict):
        full_gate_times = standard_times.get_gate_times(
            custom_gate_times=gate_times)
    elif isinstance(gate_times, GateTimes):
        full_gate_times = gate_times
    else:
        warnings.warn('Invalid gate times, assuming standard_times')
        full_gate_times = standard_times
    return full_gate_times


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

    if isinstance(T1, list):
        return thermal_relaxation_model_per_qb(T1, T2, gate_times)

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

    # Group single qubit gates into groups with the same duration
    time_group = {}
    for k, v in gate_times.get_single_qubit_times().items():
        time_group[v] = time_group.get(v, []) + [k]

    # Add the error groups to the noise model as one
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

    # Include delay and treat it as normal gates
    error_delay = thermal_relaxation_error(T1, T2, gate_times['delay'])
    noise_damping.add_all_qubit_quantum_error(error_delay,
                                              "delay")

    # Delay divided into smaller fractions
    max_fraction = 10
    for n in range(2, max_fraction+1):
        error_delay = thermal_relaxation_error(T1, T2, gate_times['delay']/n)
        noise_damping.add_all_qubit_quantum_error(error_delay,
                                                  "delay/"+str(n))
    noise_damping.add_basis_gates(['unitary'])
    return noise_damping


def thermal_relaxation_model_per_qb(T1, T2, gate_times=WACQT_gate_times):
    """Noise model for thermal relaxation. All times are given
    in nanoseconds (ns).

    Args:
        T1 (list[int], optional): List of relaxation time (ns) for each qubit in order.
        T2 (list[int], optional): List of echo dephasing time (ns) for each qubit in order.
        gate_times (dict/GateTimes object, optional): Gate times, defaults to standard WACQT_gate_times.

    Returns:
        Noise model: thermal relaxation noise model
    """

    if len(T1) != len(T2):
        raise ValueError("T1 and T2 lists are not of the same length")

    # Convert from dict object to GateTimes object
    if isinstance(gate_times, dict):
        gate_times = GateTimes(
            0, 0, WACQT_gate_times.get_gate_times(gate_times))

    noise_damping = NoiseModel()

    # Group single qubit gates into groups with the same duration
    time_group = {}
    for k, v in gate_times.get_single_qubit_times().items():
        time_group[v] = time_group.get(v, []) + [k]

    # Add the groups to the model as one
    for i, (T1_tmp, T2_tmp) in enumerate(zip(T1, T2)):
        for t_single in time_group:
            error_single = thermal_relaxation_error(T1_tmp, T2_tmp, t_single)
            noise_damping.add_quantum_error(error_single,
                                            time_group[t_single],
                                            [i])

    # Do the same for two qubit gates
    time_group = {}
    for k, v in gate_times.get_two_qubit_times().items():
        time_group[v] = time_group.get(v, []) + [k]

    for i, (T1_first_qubit, T2_first_qubit) in enumerate(zip(T1, T2)):
        for j, (T1_second_qubit, T2_second_qubit) in enumerate(zip(T1, T2)):
            for t_double in time_group:
                error_double = thermal_relaxation_error(T1_first_qubit, T2_first_qubit, t_double).expand(
                    thermal_relaxation_error(T1_second_qubit, T2_second_qubit, t_double))
                noise_damping.add_quantum_error(error_double,
                                                time_group[t_double],
                                                [i, j])

    # NOTE: More consistent to loop over special_ops?
    for i, (T1_tmp, T2_tmp) in enumerate(zip(T1, T2)):
        error_reset = thermal_relaxation_error(
            T1_tmp, T2_tmp, gate_times['reset'])
        noise_damping.add_quantum_error(error_reset, "reset", [i])
        error_measure = thermal_relaxation_error(
            T1_tmp, T2_tmp, gate_times['measure'])
        noise_damping.add_quantum_error(error_measure, "measure", [i])

    for i, (T1_tmp, T2_tmp) in enumerate(zip(T1, T2)):
        # Include delay and treat it as normal gates
        error_delay = thermal_relaxation_error(
            T1_tmp, T2_tmp, gate_times['delay'])
        noise_damping.add_quantum_error(error_delay, "delay", [i])

        # Delay divided into smaller fractions
        max_fraction = 10
        for n in range(2, max_fraction+1):
            error_delay = thermal_relaxation_error(
                T1_tmp, T2_tmp, gate_times['delay']/n)
            noise_damping.add_quantum_error(error_delay, "delay/"+str(n), [i])
    noise_damping.add_basis_gates(['unitary'])

    return noise_damping

def pauli_noise_model(p_gate1=0.0, p_meas=0.0, p_reset=0.0):
    """Testing around with some Bit-flip noise"""

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

def phase_amplitude_model(T1=40e3, T2=60e3, t_single=15, t_cz=300,
                          t_measure=1000, t_reset=1000):
    """Noise model for amplitude and phase damping. All times are given
    in nanoseconds (ns) This is equivalent to the thermal_relaxation model.
    """
    warnings.warn('DEPRECATED, use thermal_relaxation_model_V2 instead', DeprecationWarning)
    
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


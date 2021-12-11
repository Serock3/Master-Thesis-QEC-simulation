# Master_thesis_QEC_simulation
Authors: Alexander Andersson and Sebastian Holmin

This repository contains the code for simulating quantum error correction (QEC), primarily for the [[5,1,3]] code, using IBM's Qiskit.
The project was in the form of a Master's thesis [1] as well as a six-week continuation of the project.

## Introduction
Quantum computers are much noisier than their classical counterparts. To realize the many proposed quantum algorithms believed to grant substantial speedups, this noise has to be suppressed through quantum error correction. This is done by encoding the information of *logical* qubits into a larger amount of *physical* qubits, creating redundant information. 
By entangling the physical qubits with extra *ancillary* ones, so-called *stabilizers* observables can be measured without affecting the logical qubit.
The measurement results give information about any error that has occurred, which can be used to apply a correction procedure mapping the qubits back to their initial state.
There exists a large variety of stabilizer codes. In this project, we explore the [[5,1,3]] code, which is the smallest that can correct *any* single-qubit error. If two or more occur, the errors cannot be uniquely identified.
The main obstacle for QEC is that the procedure is itself noisy, and will only produce an overall net reduction in errors if the substituent physical qubits are stable enough. 

This project is (mainly) limited to a device with seven qubits laid out in a hexagonal pattern with triangular gate connectivity. This graph is well suited for the [[5,1,3]] code as the center qubit has connectivity with every other qubit. Every stabilizer requires the ancillary qubit to be entangled with four data qubits. Thus, to realize the [[5,1,3]] code, we measure the stabilizers in a series, reusing the center qubit as the ancilla each time. Errors are then likely to occur between the stabilizers, and the syndromes may be incorrect.


## Project Status
**WIP**
For a thorough rundown of the results and underlying theory of this project, refer to the thesis report [1]. 


## How to use
All code is written in Python (or Jupyter Notebooks), and simulations are based on the Qiskit module by IBM (see [2] for installation). 
Here, we will not go into detail on how to use Qiskit, nor the principles of quantum computing and QEC.
For an introduction to Qiskit, refer to their excellent tutorials[2].
On the topic of quantum computing, the relevant theory is summarized in [1], for more robust literature on the topic, refer to 'Quantum Computation and Quantum Information' by M. A. Nielsen and I. L. Chuang [3]

To get started with this repository, it is recommended to start through one of the notebooks and work your way into its dependencies/imports and their use there.
A couple of examples of where to start could be:
- *422_code_summary.ipynb* introduces the main concepts in a somewhat logical fashion, although is rather disconnected from many parts of the code, as it details the [[4,2,2]] code, not [[5,1,3]].
- *active_qec.ipynb, comparing_QEC_setups.ipynb* and *gate_times_sweep.ipynb* details the simulation and visualization of results in the thesis report[1], and could be used alongside it.
- *decoding_errors.ipynb* describes the topics mostly worked on after the report and can be used to get into the unanswered or most relevant questions in this project.

The code is primarily centered around the [[5,1,3]] QEC code and its primary functions can be divided into three parts:
1. **Building the circuits.** This includes constructing specific circuits, both to contain all necessary gates and instructions, but also to compile it to a specific layout 
(i.e. not all qubits can interact with each other.)
This functionality is generally contained within the following files:
* simulator_program/stabilizers.py - Functions to create a stabilizer circuit, including encoding of the state and snapshots of intermediate results.
* simulator_program/decay.py - Functions for simulating the destruction of a logical state over time, without any error correction.
* simulator_program/custom_transpiler.py - Functions for compiling circuits to certain qubit connectivity and a limited set of gates. 
2. **Adding noise to the circuit.** Functions for adding any noise to otherwise ideal (noise-free) circuits and simulations.
This functionality is generally contained within the following files:
* simulator_program/custom_noise_model.py - Various functions for specifying the noise model. Throughout the project, thermal relaxation has been primarily used.
* simulator_program/idle_noise.py - In Qiskit qubit states are perfectly maintained while no gate is applied to it (idling). This contains functions for adding noise to qubits that are idling. 
3. **Running the circuits.** Different ways of simulating the circuit, saving the results, or combining this with the first two parts.
This functionality is generally contained within the following files:
* simulator_program/data_analysis_tools.py - Has functions for running simulations with standard settings, saving results or checking if results are already saved.
Scripts or notebooks in the main folder generally call these functions for their simulations.
4. **Processing the results.** Different functions for processing the results after a simulation. Plotting or other visualizations are instead handled on a case-by-case basis.
This functionality is generally contained within the following files:
* simulator_program/post_select - Tools for splitting up results based on measurement outcomes. Generally referred to as discarding runs which gave unwanted results.
* simulator_program/post_process - Tools for processing the density matrix of the qubit state after simulation. This can be seen as an alternative to active error correction;
instead of correcting the errors in real-time, they are tracked and a full correction is applied afterward.

Not all files are listed. Here, we gave a brief overview of the most important functionality. Further details are given in the respective files.

## Repository structure
The repository is structured to contain a set of script files in the main folder, primarily for generating data and visualizing the results

### /data
Contains any data files (.dat or .npy) of simulation results used for the thesis report. As high accuracy simulations are computationally heavy, these can be loaded in their respective scripts, instead of rerunning simulation.

### /simulator_program
Contains general functions used across different script files. These should not be run on their own, except for a small demonstration code at the bottom of the files.

### /split_data
Data files are used for later runs of so-called 'splitting circuits'. Results of this data can be seen in split_testing.py or decoding_errors.ipynb.

### /trash
Old files which are unused, rewritten, or deprecated. These scripts are not necessarily runnable as the code is not maintained. Additionally, their import statements still assume they are placed in the main folder.

## Sources
[1] Masters thesis report: https://hdl.handle.net/20.500.12380/302690

[2] Qiskit installation: https://qiskit.org/documentation/getting_started.html

[3] M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information, 10th ed. Cambridge: Cambridge University Press, 2010. ISBN 9780511976667

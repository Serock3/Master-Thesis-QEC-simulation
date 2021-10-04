# %%
from surface_code.surface_code.circuits import SurfaceCode
import sys
from qiskit import execute, Aer

simulator = Aer.get_backend('qasm_simulator')

# Set up a d=3, T=3 code
code = SurfaceCode(3, 3)
job = execute(code.circuit['0'], simulator)

raw_results = {}
raw_results['0'] = job.result().get_counts()

processed_results = {}
processed_results = code.process_results(raw_results['0'])

j = 0
for i in (processed_results):
    print("results from circuit execution round", j, ":", processed_results[2])
    j = j+1

nodesX, nodesZ = code.extract_nodes(processed_results)
print("error nodes in X", nodesX)
print("error nodes in Z", nodesZ)
print("No Z error as logical 0 state is an eigenstate of logical Z (given no noise is added to the system)")
# %%

# %%
import qiskit
from qiskit.providers.aer.library import save_density_matrix
circ = qiskit.QuantumCircuit(1, 1)
# circ.x(0)
num_measurements = 64*1-1
for _ in range(num_measurements):
    circ.measure_all()
circ.save_density_matrix([0], conditional = True)

results = qiskit.execute(circ, qiskit.Aer.get_backend('aer_simulator')).result()

for k in results.data()['density_matrix'].keys():
    print(k)
    print(len(k))
    print(int((num_measurements)/64)*16+3)
# %%

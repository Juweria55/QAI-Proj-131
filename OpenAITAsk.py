#!/usr/bin/env python
# coding: utf-8

# # Noiseless Simulator

# In[6]:


import numpy as np
from qiskit import QuantumCircuit
from qiskit import Aer, transpile, execute
from qiskit.tools.visualization import plot_histogram


# In[7]:


# create GHZ circuit

n_qubits = 4
circuit = QuantumCircuit(n_qubits)
 
circuit.h(0)
    
for qubit in range(n_qubits - 1):
    circuit.cx(qubit, qubit + 1)
    
circuit.measure_all()
circuit.draw(output='mpl')


# In[8]:


from qiskit_aer import AerSimulator
simulator = Aer.get_backend('aer_simulator')
result = simulator.run(circuit).result()
plot_histogram(result.get_counts(0))


# In[ ]:





# # Simulator with Noise

# In[22]:


import numpy as np
from qiskit import QuantumCircuit
from qiskit import Aer, transpile, execute
from qiskit.tools.visualization import plot_histogram


# In[23]:


def create_quantum_simulator():
    n_qubits = 4
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
       
    for qubit in range(n_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    
        circuit.measure_all()
        
        return(circuit)


# In[24]:


from qiskit_aer.noise import pauli_error

noise_model = NoiseModel()

p_error = 0.00017675

bit_flip = pauli_error([('X', p_error), ('I', 1 - p_error)])

print(bit_flip)


# In[25]:


# Create noisy simulator backend
sim_noise = AerSimulator(noise_model=noise_bit_flip)
 
# Transpile circuit for noisy basis gates
circ_tnoise = transpile(circ, sim_noise)
 
# Run and get counts
result_bit_flip = sim_noise.run(circ_tnoise).result()
counts_bit_flip = result_bit_flip.get_counts(0)
 
# Plot noisy output
plot_histogram(counts_bit_flip)


# In[ ]:





# In[ ]:





# # Noise Model

# In[27]:


from qiskit_aer.noise import NoiseModel
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error

noise_model = NoiseModel()

# Add depolarizing error to all single qubit u1, u2, u3 gates
error = depolarizing_error(0.1, 1)
noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
 
# Print noise model info
print(noise_model)


# In[28]:


n_qubits = 4
circ = QuantumCircuit(n_qubits)
 
circ.h(0)
for qubit in range(n_qubits - 1):
    circ.cx(qubit, qubit + 1)
circ.measure_all()
print(circ)


# In[29]:


#values from the IBM_Perth
p_reset = 0.025 
p_meas = 0.0286
p_gate1 = 0.00017
 
# QuantumError objects
error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)
 
# Add errors to noise model
noise_bit_flip = NoiseModel()
noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])
 
print(noise_bit_flip)


# In[30]:


# Create noisy simulator backend
sim_noise = AerSimulator(noise_model=noise_bit_flip)
 
# Transpile circuit for noisy basis gates
circ_tnoise = transpile(circ, sim_noise)
 
# Run and get counts
result_bit_flip = sim_noise.run(circ_tnoise).result()
counts_bit_flip = result_bit_flip.get_counts(0)
 
# Plot noisy output
plot_histogram(counts_bit_flip)


# In[ ]:





# In[ ]:





# Noise Model2

# In[270]:


import numpy as np
from qiskit import QuantumCircuit
from qiskit import Aer, transpile, execute
from qiskit.tools.visualization import plot_histogram
from qiskit.quantum_info import state_fidelity


# In[271]:


# System Specification
n_qubits = 4
circ = QuantumCircuit(n_qubits)
 
# Test Circuit
circ.h(0)
for qubit in range(n_qubits - 1):
    circ.cx(qubit, qubit + 1)
circ.measure_all()
circ.draw(output='mpl')


# In[272]:


# T1 and T2 values for qubits 0-3
T1s = np.random.normal(84.00,162.85,202.68) # Sampled from normal distribution mean 50 microsec
T2s = np.random.normal(51.54,111.42,252.84)  # Sampled from normal distribution mean 50 microsec
# manila backend
# Truncate random T2s <= T1s
T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(4)])
 
# Instruction times (in nanoseconds)
time_u1 = 0   # virtual gate
time_u2 = 50  # (single X90 pulse)
time_u3 = 100 # (two X90 pulses)
time_cx = 300
time_reset = 1000  # 1 microsecond
time_measure = 1000 # 1 microsecond
 
# QuantumError objects
errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                for t1, t2 in zip(T1s, T2s)]
errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                  for t1, t2 in zip(T1s, T2s)]
errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
              for t1, t2 in zip(T1s, T2s)]
errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
              for t1, t2 in zip(T1s, T2s)]
errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
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
 
print(noise_thermal)


# In[273]:


# Run the noisy simulation
sim_thermal = AerSimulator(noise_model=noise_thermal)
 
# Transpile circuit for noisy basis gates
circ_tthermal = transpile(circ, sim_thermal)
 
# Run and get counts
result_thermal = sim_thermal.run(circ_tthermal).result()
counts_thermal = result_thermal.get_counts(0)

# Plot noisy output
plot_histogram(counts_thermal)


# In[274]:


counts_Fidelity = counts_thermal
counts_Fidelity.append(count['0000']+counts['1111']/2**10)


# In[ ]:





# In[ ]:





# In[ ]:





# # Fake Backend

# In[254]:


from qiskit.providers.ibmq import IBMQ
from qiskit.providers.aer import AerSimulator

# get a real backend from a real provider
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_manila')

# generate a simulator that mimics the real quantum system with the latest calibration results
backend_sim = AerSimulator.from_backend(backend)


# In[ ]:





# In[ ]:





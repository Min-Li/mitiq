from typing import Tuple

import cirq
import numpy as np

from mitiq import Executor
from mitiq.shadows.rotation_gates import generate_random_pauli_strings, get_rotated_circuits


# Stage 1 of Classical Shadows: Measurements
def shadow_measure_with_executor(
        circuit: cirq.Circuit,
        executor: Executor,
        n_total_measurements: int, ) -> Tuple[np.ndarray, np.ndarray]:
    """
    
    Given a circuit, calculate the classical shadow of the circuit of aa bit
      string, and the index of Pauli matrices that were measured.

    Args: circuit (cirq.Circuit): Cirq circuit.
          n_total_measurements (int): number of snapshots.

    Returns: outcomes (array): Tuple of two numpy arrays. The first array contains measurement outcomes (-1, 1)
        while the second array contains the index for the sampled Pauli's (0,1,2=X,Y,Z).
        Each row of the arrays corresponds to a distinct snapshot or sample while each
        column corresponds to a different qubit.
    """

    # Generate random Pauli unitaries
    qubits = list(circuit.all_qubits())
    num_qubits = len(qubits)
    pauli_strings = generate_random_pauli_strings(num_qubits, n_total_measurements)
    # Attach measurement gates to the circuit
    rotated_circuits = get_rotated_circuits(circuit, pauli_strings)

    # Run the circuits to collect the outcomes
    results = executor.evaluate(rotated_circuits)

    # Transform the outcomes into a numpy array.
    shadow_outcomes = []
    for result in results:
        bitstring = list(result.get_counts().keys())[0]
        outcome = [1 - int(i) * 2 for i in bitstring]
        shadow_outcomes.append(outcome)
    # Combine the computational basis outcomes $$|\mathbf{b}\rangle$$ and the unitaries sampled from $$CL_2^{\otimes n}$$.
    shadow_outcomes = np.array(shadow_outcomes, dtype=int)
    assert shadow_outcomes.shape == (n_total_measurements, num_qubits), f"shape is {shadow_outcomes.shape}"
    pauli_strings = np.array(pauli_strings, dtype=str)
    return shadow_outcomes, pauli_strings

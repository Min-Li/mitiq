from typing import Tuple, Callable, Dict, Any, List, Union

import cirq
import numpy as np
from numpy.typing import NDArray
from qiskit_aer import Aer
from tqdm.auto import tqdm

from mitiq import MeasurementResult
from mitiq.interface.mitiq_cirq.cirq_utils import (
    sample_bitstrings as cirq_sample_bitstrings,
)
from mitiq.interface.mitiq_qiskit.conversions import to_qiskit
from mitiq.interface.mitiq_qiskit.qiskit_utils import (
    sample_bitstrings as qiskit_sample_bitstrings,
)


# Functions to implement random Pauli measurements.


def generate_random_pauli_strings(
    num_qubits: int, num_strings: int
) -> List[str]:
    """Generate a list of random Pauli strings.

    Args:
        num_qubits: The number of qubits in the Pauli strings.
        num_strings: The number of Pauli strings to generate.

    Returns:
        A list of random Pauli strings.
    """

    # Sample random Pauli operators uniformly from X, Y, Z
    unitary_ensemble = ["X", "Y", "Z"]
    paulis = np.random.choice(unitary_ensemble, (num_strings, num_qubits))
    return ["".join(pauli) for pauli in paulis]


def get_rotated_circuits(
    circuit: cirq.Circuit,
    pauli_strings: List[str],
    add_measurements: bool = True,
) -> List[cirq.Circuit]:
    """Returns a list of circuits that are identical to the input circuit,
    except that each one has single-qubit Clifford gates followed by
    measurement gates that are designed to measure the input
    Pauli strings in the Z basis.

    Args:
        circuit: The circuit to measure.
        pauli_strings: The Pauli strings to measure in each output circuit.
        add_measurements: Whether to add measurement gates to the circuit.

    Returns:
         The list of circuits with rotation and measurement gates appended.
    """
    qubits = list(circuit.all_qubits())
    num_qubits = len(qubits)
    rotated_circuits = []
    for pauli_string in pauli_strings:
        assert len(pauli_string) == num_qubits, (
            f"Pauli string must be same length as number of qubits, "
            f"got {len(pauli_string)} and {num_qubits}"
        )
        rotated_circuit = circuit.copy()
        for i, pauli in enumerate(pauli_string):
            qubit = qubits[i]
            # Pauli X measurement is equivalent to H plus a Z measurement
            if pauli == "X":
                rotated_circuit.append(cirq.H(qubit))
            # Pauli X measurement is equivalent to S^-1*H plus a Z measurement
            elif pauli == "Y":
                rotated_circuit.append(cirq.S(qubit) ** -1)
                rotated_circuit.append(cirq.H(qubit))
            # Pauli Z measurement
            else:
                assert (
                    pauli == "Z"
                ), f"Pauli must be X, Y, Z. Got {pauli} instead."
        if add_measurements:
            # append measurement gates
            rotated_circuit.append(cirq.measure(*qubits))
        # append to list of circuits
        rotated_circuits.append(rotated_circuit)
    return rotated_circuits


# Stage 1 of Classical Shadows: Measurements
def random_pauli_basis_measurement(
    circuit: cirq.Circuit,
    n_total_measurements: int,
    sampling_function: Union[str, Callable[..., MeasurementResult]] = "cirq",
    sampling_function_config: Dict[str, Any] = {},
) -> Tuple[NDArray[Any], NDArray[Any]]:
    r"""Given a circuit, perform random Pauli-basis measurements on
    the circuit and return the measurement outcomes & the sampled
    Pauli strings.

    Args:
         circuit: Cirq circuit.
         n_total_measurements: number of snapshots.
         sampling_function: Sampling function to use. If a string, then the
            string is used to look up a default sampling function. If a
            Callable, then the Callable is used as the sampling function.
            Defaults to `"cirq"`.
         sampling_function_config: Configuration
            for the sampling function. Defaults to {}. If sampling_function is
            a string, then this argument is ignored.

    Returns:
         outcomes: Tuple of two numpy arrays. The first array contains
            measurement outcomes (-1, 1) while the second array contains the
            index for the sampled Pauli's (0,1,2=X,Y,Z). Each row of the arrays
            corresponds to a distinct snapshot or sample while each column
            corresponds to a different qubit.
    """

    # Generate random Pauli unitaries
    qubits = list(circuit.all_qubits())
    num_qubits = len(qubits)
    pauli_strings = generate_random_pauli_strings(
        num_qubits, n_total_measurements
    )
    # Attach measurement gates to the circuit
    rotated_circuits = tqdm(
        get_rotated_circuits(
            circuit,
            pauli_strings,
        ),
        desc="Measurement",
        leave=False,
    )

    if isinstance(sampling_function, str):
        if sampling_function == "cirq":
            # Run the circuits to collect the outcomes for cirq)
            results = [
                cirq_sample_bitstrings(
                    rotated_circuit,
                    noise_level=(0,),
                    shots=1,
                    sampler=cirq.Simulator(),
                )
                for rotated_circuit in rotated_circuits
            ]
        elif sampling_function == "qiskit":
            # Run the circuits to collect the outcomes for cirq
            results = [
                qiskit_sample_bitstrings(
                    to_qiskit(rotated_circuit),
                    noise_model=None,
                    backend=Aer.get_backend("aer_simulator"),
                    shots=1,
                    measure_all=False,
                )
                for rotated_circuit in rotated_circuits
            ]
        else:
            raise ValueError(
                f"Sampling function {sampling_function} not supported"
            )
    else:
        results = [
            sampling_function(rotated_circuit, **sampling_function_config)
            for rotated_circuit in rotated_circuits
        ]

    # Transform the outcomes into a numpy array 0 -> 1, 1 -> -1.
    shadow_outcomes = []
    for result in results:
        bitstring = list(result.get_counts().keys())[0]
        outcome = [1 - int(i) * 2 for i in bitstring]
        shadow_outcomes.append(outcome)

    # output computational basis outcomes |b>
    # and the random unitaries in {X,Y,Z}.
    shadow_outcomes_np = np.asarray(shadow_outcomes, dtype=int)
    pauli_strings_np = np.asarray(pauli_strings, dtype=str)
    return shadow_outcomes_np, pauli_strings_np

# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Classical shadow estimation for quantum circuits. Based on the paper"""

from typing import Optional, Callable, List, Dict, Any

import cirq
import numpy as np
from numpy.typing import NDArray

import mitiq
from mitiq import MeasurementResult
from mitiq.shadows.quantum_processing import random_pauli_measurement
from mitiq.shadows.classical_postprocessing import (
    shadow_state_reconstruction,
    expectation_estimation_shadow,
    get_pauli_fidelity,
)
from mitiq.shadows.shadows_utils import (
    n_measurements_tomography_bound,
    n_measurements_opts_expectation_bound,
)


def pauli_twirling_calibrate(
        qubits: List[cirq.Qid],
        executor: Callable[[cirq.Circuit], MeasurementResult],
        k_calibration: Optional[int] = None,
        num_total_measurements_calibration: Optional[int] = None,
) -> Dict[str, float]:
    r"""
    This function returns the dictionary of the median of means estimation
    of Pauli fidelity: {:math:`\{'b':f_{b}\}_{b\in\{0,1\}^n}`}.

    Args:
        qubits: The qubits to measure.
        executor: The function to use to do quantum measurement.
        k_calibration: Number of groups of "median of means" used for calibration.
        num_total_measurements_calibration: Number of shots per group of
            "median of means" used for calibration.
    Returns:
        A dictionary containing the calibration outcomes.
    """
    # calibration circuit is of same qubit number with original circuit
    # qubits: List[cirq.Qid] = cirq.LineQubit.range(num_qubits)
    zero_circuit = cirq.Circuit()
    # define calibration parameters if not provided
    if k_calibration is None:
        k_calibration = 1
    if num_total_measurements_calibration is None:
        num_total_measurements_calibration = 20000
    # perform random Pauli measurement one the calibration circuit
    calibration_measurement_outcomes = random_pauli_measurement(
        zero_circuit,
        n_total_measurements=num_total_measurements_calibration,
        executor=executor,
        qubits=qubits,
    )

    # get the median of means estimation of Pauli fidelities
    f_est = get_pauli_fidelity(
        calibration_measurement_outcomes,
        k_calibration,
    )
    return f_est


def execute_with_shadows(
        circuit: cirq.Circuit,
        executor: Callable[[cirq.Circuit], MeasurementResult],
        observables: Optional[List[mitiq.PauliString]] = None,
        state_reconstruction: bool = False,
        rshadows: bool = False,
        calibration_results: Optional[Dict[str, float]] = None,
        *,
        k_shadows: Optional[int] = None,
        num_total_measurements_shadow: Optional[int] = None,
        error_rate: Optional[float] = None,
        failure_rate: Optional[float] = None,
        random_seed: Optional[int] = None,
) -> Dict[str, NDArray[Any]]:
    r"""
    Executes a circuit with classical shadows. This function can be used for
    state reconstruction or expectation value estimation of observables.

    Args:
        circuit: The circuit to execute.
        executor: The function to use to do quantum measurement.
        calibration_results: The calibration results.
        observables: The set of observables to measure. If None, the state
            will be reconstructed.
        state_reconstruction: Whether to reconstruct the state or estimate
            the expectation value of the observables.
        k_shadows: Number of groups of "median of means" used for shadow
            estimation.
        num_total_measurements_shadow: Number of shots per group of
            "median of means" used for shadow estimation.
        num_total_measurements_shadow: Total number of shots for shadow estimation.
        error_rate: Predicting all features with error rate
            :math:`\epsilon` via median of means prediction.
        failure_rate: :math:`\delta` Accurately predicting all features via
            median of means prediction with error rate less than or equals to
            :math:`\epsilon` with probability at least :math:`1 - \delta`.
        random_seed: The random seed to use for the shadow measurements.

    Returns:
        A dictionary containing the shadow outcomes, the Pauli strings, and
        either the estimated density matrix or the estimated expectation
        values of the observables.
    """

    qubits = list(circuit.all_qubits())
    num_qubits = len(qubits)

    if observables is None:
        assert (
                state_reconstruction is True
        ), "observables must be provided if state_reconstruction is False"

    if error_rate is not None:
        if state_reconstruction:
            num_total_measurements_shadow = n_measurements_tomography_bound(
                error_rate, num_qubits=num_qubits
            )
            k_shadows = 1
        else:  # Estimation expectation value of observables
            assert failure_rate is not None
            assert observables is not None and len(observables) > 0
            (
                num_total_measurements_shadow,
                k_shadows,
            ) = n_measurements_opts_expectation_bound(
                error=error_rate,
                observables=observables,
                failure_rate=failure_rate,
            )
    else:
        assert num_total_measurements_shadow is not None
        if not state_reconstruction:
            assert k_shadows is not None

    if random_seed is not None:
        np.random.seed(random_seed)

    if rshadows:
        assert calibration_results is not None

    """
    Stage 1: Shadow Measurement
    """
    shadow_outcomes, pauli_strings = random_pauli_measurement(
        circuit,
        n_total_measurements=num_total_measurements_shadow,
        executor=executor,
    )
    output = {
        "shadow_outcomes": shadow_outcomes,
        "pauli_strings": pauli_strings,
    }
    """
    Stage 2: Estimate the expectation value of the observables OR reconstruct
    the state
    """
    measurement_outcomes = (shadow_outcomes, pauli_strings)
    if state_reconstruction:
        est_density_matrix = shadow_state_reconstruction(
            measurement_outcomes,
            rshadows,
            f_est=calibration_results)
        output["est_density_matrix"] = est_density_matrix
    else:  # Estimation expectation value of observables
        assert observables is not None and len(observables) > 0
        assert k_shadows is not None
        expectation_values = [
            expectation_estimation_shadow(
                measurement_outcomes,
                obs,
                k_shadows=int(k_shadows),
                pauli_twirling_calibration=rshadows,
                f_est=calibration_results
            )
            for obs in observables
        ]
        output["est_observables"] = np.array(expectation_values)
    return output

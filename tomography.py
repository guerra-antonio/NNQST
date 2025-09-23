import numpy as np
import cvxpy as cp
import itertools
import scipy.linalg

from tqdm import tqdm
import torch
from classical_shadow import sample_measurements, NQS, nll_batch
from IPython.display import clear_output


# ------------------------------------------
# Standard - Compressed Sensing Tomography
# ------------------------------------------
def pauli_basis(N: int = 3):
    """
    Generates all non-trivial tensor products of Pauli matrices (excluding the global identity)
    for an N-qubit system. Returns a list of 4^N - 1 matrices, each of shape (2^N, 2^N).

    Args:
        N (int): Number of qubits.

    Returns:
        basis (list of np.ndarray): List of Pauli product matrices (excluding global identity).
    """
    paulis = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }

    # Generate all N-length strings over 'I', 'X', 'Y', 'Z'
    keys = ['I', 'X', 'Y', 'Z']
    strings = [''.join(s) for s in itertools.product(keys, repeat=N)]

    # Exclude the all-identity string
    strings = [s for s in strings if not all(c == 'I' for c in s)]

    basis = []
    for s in strings:
        # Compute the tensor product for this Pauli string
        op = paulis[s[0]]
        for k in s[1:]:
            op = np.kron(op, paulis[k])
        basis.append(op)
    return basis


def gellmann_basis(d: int = 8):
    """
    Generates the full set of generalized Gell-Mann matrices for Hilbert space dimension d.
    These (d^2 - 1) matrices are traceless, Hermitian, and form an orthonormal basis
    for the space of traceless Hermitian operators (su(d)).

    Args:
        d (int): Hilbert space dimension (default 8).

    Returns:
        gms (list of np.ndarray): List of (d^2 - 1) Gell-Mann matrices, each of shape (d, d).
    """
    gms = []

    # Symmetric and anti-symmetric off-diagonal matrices
    for j in range(d):
        for k in range(j + 1, d):

            # Symmetric: E_{jk} + E_{kj}
            S = np.zeros((d, d), dtype=complex)
            S[j, k] = 1
            S[k, j] = 1
            gms.append(S)

            # Anti-symmetric: -i E_{jk} + i E_{kj}
            A = np.zeros((d, d), dtype=complex)
            A[j, k] = -1j
            A[k, j] = 1j
            gms.append(A)

    # Diagonal traceless matrices (generalizing lambda_3, lambda_8, etc.)
    for l in range(1, d):
        D = np.zeros((d, d), dtype=complex)
        factor = np.sqrt(2 / (l * (l + 1)))

        # Set first l diagonal entries to +1
        for j in range(l):
            D[j, j] = 1

        # Set (l, l) entry to -l to ensure tracelessness
        D[l, l] = -l
        D *= factor
        gms.append(D)

    return gms


def simulate_expectation(
    rho: np.ndarray,
    O: np.ndarray,
    n_exp: int = 1000,
    seed = None,
    error : float = 0.0
):
    rng = np.random.default_rng(seed)

    # Diagonalize observable: O = V D V†
    eigvals, eigvecs = np.linalg.eigh(O)

    # Calculate probabilities for each eigenvector (projector)
    if error != 0.0:
        rho = (1-error) * rho + error / 2 * np.eye(2, dtype = np.complex64)
        
    probs = np.array([np.real(np.trace(rho @ np.outer(v, v.conj()))) for v in eigvecs.T])

    # Normalize in case of rounding error
    probs = np.maximum(probs, 0)
    probs /= np.sum(probs)

    # Simulate n_exp measurement outcomes
    outcomes = rng.choice(eigvals, size=n_exp, p=probs)
    unique_vals, counts = np.unique(outcomes, return_counts=True)
    exp_val_estimate = np.sum(unique_vals * counts) / n_exp
    return exp_val_estimate


def compressed_sensing(
    expectation_values: np.ndarray,
    set_measure_basis: list,
    epsilon: float = 1e-2
):
    """
    Quantum compressed sensing state tomography via nuclear norm minimization.

    Args:
        expectation_values (np.ndarray): Array of measured expectation values (shape: [m]).
        set_measure_basis (list): List of observables (e.g. Pauli or Gell-Mann matrices), length m.
        epsilon (float, optional): Allowed numerical tolerance for constraint satisfaction (default: 1e-6).

    Returns:
        np.ndarray: The reconstructed density matrix (dim x dim), physically valid.
    """
    dim = set_measure_basis[0].shape[0]
    n_Obs = expectation_values.shape[0]
    assert n_Obs == len(set_measure_basis), "Mismatch: Number of expectation values and measurement bases must be equal."

    rho = cp.Variable((dim, dim), hermitian=True)
    constraints = [rho >> 0, cp.trace(rho) == 1]

    # Add measurement constraints with allowed slack (epsilon)
    for idx in range(n_Obs):
        Obs = set_measure_basis[idx]
        constraints.append(cp.abs(cp.trace(rho @ Obs) - expectation_values[idx]) <= epsilon)

    # Objective: Minimize nuclear norm (trace norm) of rho (promotes low rank)
    prob = cp.Problem(cp.Minimize(cp.normNuc(rho)), constraints)
    prob = prob.solve(solver='SCS', verbose=False)

    return rho.value



def standard_tomography(
    set_measure_basis: list,
    expectation_values: np.ndarray,
    *,
    solver: str = "SCS",
    verbose: bool = False,
    **solve_kwargs
):
    """
    Standard quantum state tomography via least squares fitting.
    Finds the physical density matrix closest (in the least squares sense) to the measured expectation values.

    If the solver fails to converge or no solution is returned, this function returns None.

    Args:
        set_measure_basis (list): List of measurement observables (Pauli, Gell-Mann, etc.).
        expectation_values (np.ndarray): Measured expectation values, in the same order.
        solver (str, optional): CVXPY solver name (default: 'SCS').
        verbose (bool, optional): Pass-through to cvxpy's `solve`.
        **solve_kwargs: Extra keyword args forwarded to `prob.solve(...)`.

    Returns:
        np.ndarray | None: Reconstructed density matrix (dim x dim), or None if no solution.
    """
    dim = set_measure_basis[0].shape[0]
    n_bases = len(set_measure_basis)

    rho = cp.Variable((dim, dim), hermitian=True)
    constraints = [rho >> 0, cp.trace(rho) == 1]

    # Objective: minimize sum of squared errors between predicted and measured values
    squared_errors = []
    for idx in range(n_bases):
        Obs = set_measure_basis[idx]
        v = expectation_values[idx]
        squared_errors.append(cp.square(cp.abs(cp.trace(rho @ Obs) - v)))

    objective = cp.Minimize(cp.sum(squared_errors))
    prob = cp.Problem(objective, constraints)

    default_kwargs = dict(
    eps=1e-6,
    max_iters=10_000,
    acceleration_lookback=20,
    use_indirect=False,     # ← prueba True y False y qué gana en ese PC
    warm_start=True)
    default_kwargs.update(solve_kwargs)

    try:
        prob.solve(solver=cp.SCS, verbose=verbose, **default_kwargs)
    except Exception:
        # Any solver/runtime error -> no solution
        return None

    # Accept only statuses that return a (possibly inexact) solution
    acceptable = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if prob.status not in acceptable or rho.value is None:
        return None

    return rho.value


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Computes the quantum fidelity F(rho, sigma) between two density matrices.
    For general (possibly mixed) states:
        F = [Tr(sqrt(sqrt(rho) * sigma * sqrt(rho)))]^2

    Robust version:
    - Returns 0.0 if either input is None (e.g., tomography didn't converge).
    - Returns 0.0 on numerical errors or non-finite results.
    - Clamps the final result to [0, 1] to mitigate tiny numerical overshoots.

    Args:
        rho (np.ndarray): Density matrix (d x d), PSD, trace 1. Or None.
        sigma (np.ndarray): Density matrix (d x d), PSD, trace 1. Or None.

    Returns:
        float: Fidelity in [0,1]; returns 0.0 if inputs are invalid or computation fails.
    """
    # If tomography failed upstream (rho or sigma is None), return 0.0
    if rho is None or sigma is None:
        return 0.0

    try:
        # Basic shape check
        if rho.shape != sigma.shape or rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            return 0.0

        # Compute Uhlmann fidelity
        sqrt_rho = scipy.linalg.sqrtm(rho)
        inner = sqrt_rho @ sigma @ sqrt_rho
        sqrt_inner = scipy.linalg.sqrtm(inner)
        F = np.trace(sqrt_inner)
        F = float(np.real(F)**2)

        # Guard against numerical issues
        if not np.isfinite(F):
            return 0.0

        # Clip to [0,1] to counter tiny negative or >1 due to roundoff
        return float(np.clip(F, 0.0, 1.0))

    except Exception:
        # Any numerical error -> return 0.0 per your requirement
        return 0.0


# ------------------------------------------
# Neural Network Quantum State Tomography
# ------------------------------------------
# Define the Neural Quantum State model, the optimizer an the scheduler
model   = NQS()

# Function to create mini-batches
def batches(lst, bs):
    """
    Yield successive mini-batches of size `bs` from a list.

    Args:
        lst: list, dataset or collection to be split into batches.
        bs: int, batch size.

    Yields:
        list, a slice of `lst` containing up to `bs` elements.
    """
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

# Function for train the model
opt     = torch.optim.Adam(model.parameters(), lr=1e-2)
sch     = torch.optim.lr_scheduler.StepLR(opt, step_size = 300, gamma=0.5)

def nn_tomography(
    train_data,
    test_data=None,
    epochs=900,
    batch=128,
):
    """
    Train the global `model` on (s_bits, B) samples using NLL (in bits).

    Args:
        train_data: list of (s_bits, B) training samples
        test_data:  optional list of (s_bits, B) for validation
        epochs:     number of training epochs
        batch:      mini-batch size

    Returns:
        model:       the trained global model
        loss_train:  list of per-epoch training NLL (bits)
        loss_test:   list of per-epoch validation NLL (bits) if test_data is provided,
                     otherwise an empty list
    """
    loss_train = []
    loss_test  = []

    for epoch in tqdm(range(1, epochs + 1), desc="Training Epochs"):
        model.train()
        epoch_losses = []

        # Iterate over mini-batches
        for batch_samples in batches(train_data, batch):
            loss = nll_batch(model, batch_samples)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(loss.item())

        # End-of-epoch bookkeeping
        train_mean = float(np.mean(epoch_losses))
        loss_train.append(train_mean)

        # Step LR scheduler once per epoch (consistent with your original code)
        sch.step()

        # Validation (optional)
        val_loss = None
        if test_data is not None:
            model.eval()
            with torch.no_grad():
                val_loss = nll_batch(model, test_data).item()
            loss_test.append(val_loss)

        # Nice progress printing
        if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
            clear_output(wait=True)
            if val_loss is not None:
                print(f"Epoch {epoch:3d} | train NLL (bits): {train_mean:.6f} | val NLL (bits): {val_loss:.6f}")
            else:
                print(f"Epoch {epoch:3d} | train NLL (bits): {train_mean:.6f}")

    return model, loss_train, loss_test
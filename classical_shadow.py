# ======================
# Import necessary libraries
# ======================
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from random import random, shuffle

# ======================
# Before start, it is useful to define a function to randomly generates an state
# ======================

def random_state():
    """
    Generate a random pure single-qubit state |ψ⟩ uniformly distributed
    on the Bloch sphere.

    The state is parametrized as:
        |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
    where θ ∈ [0, π], φ ∈ [0, 2π].

    Returns:
        torch.Tensor of shape (2,1), complex column vector representing
        the qubit state |ψ⟩.
    """
    # Generate two random numbers uniformly distributed in [0,1]
    u = torch.rand(1)
    v = torch.rand(1)

    # Polar angle θ ∈ [0, π]
    # Using the transformation θ = arccos(1 - 2u) ensures uniformity on the Bloch sphere
    theta = torch.acos(1 - 2 * u)

    # Azimuthal angle φ ∈ [0, 2π], chosen uniformly
    phi = 2 * torch.pi * v

    # Construct the qubit state:
    # |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
    psi = torch.tensor([
        torch.cos(theta / 2),
        torch.exp(1j * phi) * torch.sin(theta / 2)
    ], dtype=torch.complex64)

    # Return the state as a column vector
    return psi.reshape(-1, 1)


# ======================
# Define the theoretical probabilities per basis
# ======================
eps    = 1e-10

pauli = {
    "I": torch.tensor([[1, 0],
                       [0, 1]], dtype=torch.complex64),
    "X": torch.tensor([[0, 1],
                       [1, 0]], dtype=torch.complex64),
    "Y": torch.tensor([[0, -1j],
                       [1j, 0]], dtype=torch.complex64),
    "Z": torch.tensor([[1, 0],
                       [0, -1]], dtype=torch.complex64),
}

def probs_true_in_basis(psi: torch.Tensor, B: str, error: float = 0.0) -> torch.Tensor:
    """
    Compute measurement probabilities p(s|B) for a single-qubit state |ψ⟩
    in basis B ∈ {'X','Y','Z'}, including depolarizing noise.

    Args:
        psi: torch.Tensor of shape (2,1), complex column vector.
        B: str, measurement basis ('X','Y','Z').
        error: float in [0,1], depolarizing noise probability.
               - 0.0 → pure state
               - 1.0 → maximally mixed state (I/2)

    Returns:
        torch.Tensor of shape (2,), real probabilities [p0, p1].
            Convention:
                - s=0 → eigenvector with eigenvalue +1
                - s=1 → eigenvector with eigenvalue -1
    """
    # Normalize |ψ⟩
    psi = psi.reshape(2, 1).to(torch.complex64)
    psi = psi / torch.linalg.norm(psi)

    # Build depolarized ρ
    I = torch.eye(2, dtype=torch.complex64, device=psi.device)
    rho = (1.0 - error) * (psi @ psi.conj().T) + (error / 2.0) * I

    # Eigendecomposition of σ_B (vals = [-1, +1])
    eigvals, eigvecs = torch.linalg.eigh(pauli[B.upper()])

    v_plus  = eigvecs[:, 1].reshape(-1, 1)  # eigenvalue +1 → s=0
    v_minus = eigvecs[:, 0].reshape(-1, 1)  # eigenvalue -1 → s=1

    # Projectors
    proj0 = v_plus  @ v_plus.conj().T
    proj1 = v_minus @ v_minus.conj().T

    # Measurement probabilities
    p0 = torch.trace(rho @ proj0).real
    p1 = torch.trace(rho @ proj1).real

    # Stable clamp + normalization
    p = torch.stack((p0, p1)).clamp_min(eps)
    p = (p / p.sum()).to(torch.float32)
    return p


# ======================
# Simulate the data (s, B)
# ======================
def sample_measurements(psi, M=2000, probs_bases=(1/3, 1/3, 1/3), seed = None, error=0.0):
    """
    Generate measurement outcomes for a single-qubit quantum state |ψ⟩.

    Measurements are performed in bases {X, Y, Z} according to the probabilities 
    specified in `probs_bases`. For each chosen basis B, the true measurement 
    distribution p_true(s|B) is computed via `probs_true_in_basis` (including optional
    depolarizing noise). Then an outcome s ∈ {0,1} is sampled.

    Args:
        psi (torch.Tensor): Column vector of shape (2,1) representing the qubit state |ψ⟩.
        M (int): Number of measurement shots (default 2000).
        probs_bases (tuple): Probabilities of selecting bases (X, Y, Z). Must sum to 1.
        seed (int): Random seed for reproducibility.
        error (float): Depolarizing noise parameter in [0,1]. 
                       error=0 → noiseless; error=1 → fully mixed state.

    Returns:
        list: A list of M tuples [(s_tensor, B_char)], where
              - s_tensor is a torch.LongTensor([0]) or torch.LongTensor([1]),
              - B_char is the basis used ('X', 'Y', or 'Z').
    """
    rng = np.random.default_rng(seed)

    # Bases to choose from and their probabilities
    bases  = np.array(['X','Y','Z'])
    probsB = np.array(probs_bases, dtype=np.float64)
    # Clip small negatives, then normalize exactly to sum=1
    probsB = np.clip(probsB, 0.0, 1.0)
    probsB = probsB / probsB.sum()

    eps = 1e-12  # numerical stability cutoff
    data = []

    for _ in range(M):
        # Randomly choose a basis B according to probs_bases
        B = rng.choice(bases, p=probsB).item()

        # Compute true distribution p(s|B) as a torch tensor
        p_torch = probs_true_in_basis(psi=psi, B=B, error=error)  # shape (2,)

        # Convert to NumPy, clip and normalize to ensure sum=1
        p = p_torch.detach().to(torch.float64).cpu().numpy()
        p = np.clip(p, eps, 1.0)
        p = p / p.sum()

        # Sample measurement outcome s ∈ {0,1}
        s = int(rng.choice(2, p=p))

        # Store result as (tensor outcome, basis label)
        data.append((torch.tensor([s], dtype=torch.long), B))

    return data


# ======================
# To include errors we define a function to flips a certain percentage of the dataset.
# ======================
def flip_dataset(dataset, p: float):
    """
    Apply bit-flip noise to a fraction p of a dataset.

    Each element of the dataset is assumed to be a tuple (bit, basis), where:
        - bit: torch.Tensor of shape (1,) containing either 0 or 1
        - basis: additional information (e.g., measurement basis)

    The function flips the first int(len(dataset) * p) elements:
        - 0 → 1
        - 1 → 0
    while leaving the rest unchanged.

    Args:
        dataset: list of tuples (bit, basis), original dataset.
        p (float): fraction of elements to flip, in [0,1].

    Returns:
        list of tuples (bit, basis), modified dataset with flipped bits.
    """
    new_dataset = []  # Store the new dataset with possible flips

    elements_error = int(len(dataset) * p)  
    # Number of elements to flip (always the first `elements_error`)

    for i in range(len(dataset)):
        if i < elements_error:
            bit, basis = dataset[i]

            # Flip the bit: 0 → 1, 1 → 0
            if bit.item() == 0:
                new_bit = 1
            else:
                new_bit = 0

            # Create a new tuple with the flipped bit and the same basis
            new_tuple = (torch.tensor([new_bit]), basis)
            new_dataset.append(new_tuple)
        else:
            # Keep the element unchanged
            new_dataset.append(dataset[i])
    shuffle(new_dataset)  # Mix the dataset
    return new_dataset


# ======================
# Define the Neural Quantum State model, that is, the Neural Network model used to represent the Quantum State
# ======================
# class NQS(nn.Module):
#     """
#     Neural Quantum State (NQS) for a single qubit.

#     Parameters (learnable):
#         - logits (2,): real parameters used to define probabilities p(0), p(1).
#                        A softmax ensures automatic normalization.
#         - phases (2,): real parameters corresponding to phases φ₀ and φ₁.

#     Amplitudes:
#         α₀ = sqrt(p₀) * exp(i φ₀)
#         α₁ = sqrt(p₁) * exp(i φ₁)
#     """

#     def __init__(self):
#         super().__init__()
#         self.logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))  # probabilities
#         self.phases = nn.Parameter(torch.zeros(2, dtype=torch.float32))  # phases

#     def alphas(self):
#         """
#         Compute the complex amplitudes α = (α₀, α₁) and probabilities p = (p₀, p₁).

#         Returns:
#             amp: torch.Tensor of shape (2,), complex amplitudes.
#             probs: torch.Tensor of shape (2,), real probabilities (softmax-normalized).
#         """
#         probs = F.softmax(self.logits, dim=0)  # normalized probabilities
#         amp   = torch.sqrt(probs) * torch.exp(1j*self.phases.to(torch.float32))
#         return amp, probs

#     def prob_s_given_B(self, s, B):
#         """
#         Compute model probability p_λ(s|B) of outcome s ∈ {0,1}
#         when measuring in basis B ∈ {'X','Y','Z'}.

#         Args:
#             s: torch.LongTensor of shape (1,), outcome (0 or 1).
#             B: str, measurement basis ('X','Y','Z').

#         Returns:
#             torch.Tensor (scalar), probability p_λ(s|B).
#         """
#         al, probs = self.alphas()
#         a0, a1 = al[0], al[1]

#         if B == 'Z':
#             # In Z basis, probability is directly |α_s|² = p(s)
#             return probs[s]

#         elif B == 'X':
#             # In X basis: |±⟩ = (|0⟩ ± |1⟩)/√2
#             A = (a0 + ((-1)**s.item()) * a1) / np.sqrt(2.0)
#             return (A.conj()*A).real.clamp_min(eps)

#         elif B == 'Y':
#             # In Y basis: |+i⟩ = (|0⟩ + i|1⟩)/√2, |−i⟩ = (|0⟩ − i|1⟩)/√2
#             A = (a0 + (1j if s.item() == 0 else -1j) * a1) / np.sqrt(2.0)
#             return (A.conj()*A).real.clamp_min(eps)

#         else:
#             raise ValueError("Basis must be 'X','Y','Z'")

#     def statevector(self):
#         """
#         Construct the normalized statevector |ψ⟩ corresponding to the NQS.

#         Returns:
#             psi: torch.Tensor of shape (2,1), complex column vector.
#         """
#         al, _ = self.alphas()
#         psi = al.reshape(2,1).to(torch.complex64)

#         # Normalize the state for numerical safety
#         psi = psi / torch.linalg.norm(psi)
#         return psi

class NQS(nn.Module):
    """
    Neural Quantum State (NQS) for a single qubit.

    Parameters (learnable):
        - logits (2,): real parameters used to define probabilities p(0), p(1).
                       A softmax ensures automatic normalization.
        - phases (2,): real parameters corresponding to phases φ₀ and φ₁.

    Amplitudes:
        α₀ = sqrt(p₀) * exp(i φ₀)
        α₁ = sqrt(p₁) * exp(i φ₁)
    """

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))  # probabilities
        self.phases = nn.Parameter(torch.zeros(2, dtype=torch.float32))  # phases

    def alphas(self):
        """
        Compute the complex amplitudes α = (α₀, α₁) and probabilities p = (p₀, p₁).

        Returns:
            amp: torch.Tensor of shape (2,), complex amplitudes.
            probs: torch.Tensor of shape (2,), real probabilities (softmax-normalized).
        """
        probs = F.softmax(self.logits, dim=0)  # normalized probabilities
        amp   = torch.sqrt(probs) * torch.exp(1j*self.phases.to(torch.float32))
        return amp, probs

    def psi(self):
        """
        Construct the normalized statevector |ψ⟩ corresponding to the NQS.

        Returns:
            psi: torch.Tensor of shape (2,1), complex column vector.
        """
        al, _ = self.alphas()
        psi = al.reshape(2,1).to(torch.complex64)

        # Normalize the state for numerical safety
        psi = psi / torch.linalg.norm(psi)
        return psi

    def rho(self):
        psi = self.psi()                  # asumo que al es un tensor de tamaño (2,) para 1 qubit, o (2**n,)
        rho = torch.kron(psi, psi.conj().T)               # |psi><psi|
        # seguridad numérica: fuerza traza 1 si hay redondeos
        rho = rho / torch.real(torch.trace(rho))
        return rho
    
    def probs_in_basis(self, B: str) -> torch.Tensor:
        """
        Devuelve [p0, p1] con autovectores de σ_B.
        s=0 -> autovalor +1, s=1 -> autovalor -1.
        """
        B = B.upper()
        if B not in pauli:
            raise ValueError("Basis must be 'X','Y','Z'")

        psi = self.psi()
        sigmaB = pauli[B].to(psi.device)

        # Autodescomposición de σ_B (Hermítica): vals = [-1, +1]
        eigvals, eigvecs = torch.linalg.eigh(sigmaB)

        # Índices: +1 es la columna 1, -1 es la columna 0
        v_plus  = eigvecs[:, 1].reshape(-1, 1)  # autovalor +1 -> s=0
        v_minus = eigvecs[:, 0].reshape(-1, 1)  # autovalor -1 -> s=1

        # Proyectores Π_s = |v_s><v_s|
        proj0 = v_plus  @ v_plus.conj().T
        proj1 = v_minus @ v_minus.conj().T

        p0 = psi.conj().T @ proj0 @ psi
        p1 = psi.conj().T @ proj1 @ psi

        p = torch.stack([p0.real, p1.real]).clamp_min(eps)
        p = (p / p.sum()).to(torch.float32)
        return p
        # return print(p0.shape), print(p1.shape)
        # return print(proj0.shape), print(psi.shape)

    def prob_s_given_B(self, s, B: str) -> torch.Tensor:
        if isinstance(s, torch.Tensor):
            s = int(s.item())
        if s not in (0, 1):
            raise ValueError("s must be 0 or 1")
        return self.probs_in_basis(B)[s]


# ======================
# Loss function: Negative Log-Likelihood (NLL) over (s, B) samples. Also, we define an useful function to prepare the different batches used in the optimizing step
# ======================
def nll_batch(model, batch):
    """
    Compute the negative log-likelihood (NLL) loss for a batch of 
    measurement outcomes.

    Each element in the batch is a tuple (s, B), where:
        - s: torch.LongTensor of shape (1,), outcome (0 or 1).
        - B: str, measurement basis ('X','Y','Z').

    The model provides probabilities p_λ(s|B). The loss is defined as:

        NLL = - (1/N) Σ log₂( p_λ(s|B) )

    where the average is taken over all samples in the batch.

    Args:
        model: instance of NQS (or similar), implementing prob_s_given_B.
        batch: list of (s, B) pairs.

    Returns:
        torch.Tensor (scalar), the negative log-likelihood.
    """
    logps = []
    for s, B in batch:
        p = model.prob_s_given_B(s, B)   # probability p_λ(s|B)
        logps.append(torch.log2(p))      # log-likelihood contribution
    
    logps = sum(logps) / len(logps)      # average log-likelihood
    return -logps                        # negative log-likelihood

def batches(lst, bs):
    """
    Yield successive mini-batches of size `bs` from a list.

    Args:
        lst: list, the dataset to be split into batches.
        bs: int, batch size.

    Yields:
        list, a slice of `lst` containing up to `bs` elements.
    """
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]
# Importamos librerias necesarias
import numpy as np

import torch

from tqdm import tqdm
from classical_shadow import sample_measurements, NQS, batches, nll_batch, random_state
from tomography import pauli_basis, simulate_expectation, standard_tomography
from useful_functions import fidelity


# ----------------------------------------
# For test the tomography it is necessary to repeatedly make the tomography for a large amount of states with different tomography-paramenters, that is, different error and number of experiments. For that, we define a function that receives an state, a n_exp and an error percentage to include in the analysis.
# ----------------------------------------
def CS_tomography(psi_true, error: float = 0.0, n_exp: int = 500, epochs: int = 500, batch: int = 64):
    """
    Perform compressed sensing (CS) quantum state tomography using
    a Neural Quantum State (NQS) model.

    The procedure:
        1. Generate a dataset of measurement outcomes from the true state |ψ⟩.
        2. Corrupt a fraction `error` of the dataset by applying bit flips.
        3. Train the NQS model to reconstruct the state by minimizing
           the negative log-likelihood of the observed data.

    Args:
        psi_true: torch.Tensor, the true quantum state |ψ⟩ (column vector).
        error (float, optional): fraction of flipped measurements to simulate noise.
        n_exp (int, optional): number of measurements to generate for training.
        epochs (int, optional): number of training epochs.
        batch (int, optional): batch size for stochastic training.

    Returns:
        model: trained NQS model approximating the true state |ψ⟩.
    """
    # Step 1: generate a training dataset of measurement outcomes
    train_data = sample_measurements(psi = psi_true, M=n_exp, error = error)

    # # Step 2: introduce noise by flipping a fraction of the dataset
    # train_data = flip_dataset(dataset=train_data, p=error)
    
    # Step 3: prepare the model, optimizer, and learning rate scheduler
    model = NQS()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5)

    # Step 4: training loop
    for epoch in range(1, epochs + 1):
        model.train()  # set model to training mode
    
        for batch_data in batches(train_data, batch):
            # Compute negative log-likelihood loss for the current batch
            loss = nll_batch(model, batch_data)

            # Gradient step
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Update learning rate schedule
        sch.step()

    return model


# ----------------------------------------
# With the previous function already defined, we generates a set of N = 100 random quantum states and made the tomography using Classical Shadows and Standard tomography
# ----------------------------------------

# Prepare a function to compute the fidelity between the target state with the model
def fidelity_model(model, psi_true):
    """
    Compute the fidelity between the true quantum state |ψ_true⟩
    and the state |ψ_est⟩ estimated by a model.

    Fidelity is defined as:
        F = |⟨ψ_true | ψ_est⟩|^2

    Args:
        model: a trained model with a `.statevector()` method that returns
               the estimated state |ψ_est⟩ as a torch.Tensor of shape (d,1).
        psi_true: torch.Tensor of shape (d,1), the true state |ψ_true⟩.

    Returns:
        float: fidelity value F ∈ [0,1].
    """
    # Get the estimated state vector from the model
    rho_est = model.rho()

    # Compute the density matrices related
    rho_true = torch.kron(psi_true, psi_true.conj().T)

    # Compute the inner product ⟨ψ_true|ψ_est⟩
    F = fidelity(rho = rho_true, sigma = rho_est, lib = "torch")

    return F.item()

# Prepare the dataframe of Pandas to save the datagenerated

N_exp_test  = [25 * _ for _ in range(1, 20)]
p_basis     = pauli_basis(N = 1)

for n_exp in tqdm(N_exp_test, desc = "Generating data for comparisson"):
    # Prepare a list to save the results
    fidelities = [], [], [], [], [], [], [], [], [], []

    # We repeat the same process multiple times using different randomly generated quantum states
    for _ in range(100):
        # Choose a random quantum state
        psi_true = random_state()

        # -------------------------------
        # Made a set of 4 tomographies using Classical Shadow with error = 0.01, 0.05, 0.1 and 0.2
        # -------------------------------
        CSpsi_pred_0   = CS_tomography(psi_true = psi_true, error = 0.0, n_exp = n_exp)
        CSpsi_pred_001 = CS_tomography(psi_true = psi_true, error = 0.01, n_exp = n_exp)
        CSpsi_pred_005 = CS_tomography(psi_true = psi_true, error = 0.05, n_exp = n_exp)
        CSpsi_pred_01  = CS_tomography(psi_true = psi_true, error = 0.1, n_exp = n_exp)
        CSpsi_pred_02  = CS_tomography(psi_true = psi_true, error = 0.2, n_exp = n_exp)

        # Compute the fidelities for each prediction
        CSfidelity_0   = fidelity_model(model = CSpsi_pred_0, psi_true = psi_true)
        CSfidelity_001 = fidelity_model(model = CSpsi_pred_001, psi_true = psi_true)
        CSfidelity_005 = fidelity_model(model = CSpsi_pred_005, psi_true = psi_true)
        CSfidelity_01  = fidelity_model(model = CSpsi_pred_01, psi_true = psi_true)
        CSfidelity_02  = fidelity_model(model = CSpsi_pred_02, psi_true = psi_true)

        # -------------------------------
        # Then, we do two tomographies more using standard tomograhy, one using Pauli basis and the other using Gellmann matrices
        # -------------------------------
        # Prepare the proper density matrices related with the psi_true vector
        rho_true = psi_true.detach().numpy()
        rho_true = np.kron(rho_true.conj().T, rho_true).reshape(2, 2)

        # Simulate the different expectations with different errors.
        pauli_expectation_0   = np.array([simulate_expectation(rho = rho_true, O = pauli_obs, n_exp = int(n_exp / 3), error = 0.0) for pauli_obs in p_basis])
        pauli_expectation_001 = np.array([simulate_expectation(rho = rho_true, O = pauli_obs, n_exp = int(n_exp / 3), error = 0.01) for pauli_obs in p_basis])
        pauli_expectation_005 = np.array([simulate_expectation(rho = rho_true, O = pauli_obs, n_exp = int(n_exp / 3), error = 0.05) for pauli_obs in p_basis])
        pauli_expectation_01  = np.array([simulate_expectation(rho = rho_true, O = pauli_obs, n_exp = int(n_exp / 3), error = 0.1) for pauli_obs in p_basis])
        pauli_expectation_02  = np.array([simulate_expectation(rho = rho_true, O = pauli_obs, n_exp = int(n_exp / 3), error = 0.2) for pauli_obs in p_basis])

        # Then we do the standard tomography
        rho_standard_0   = standard_tomography(set_measure_basis = p_basis, expectation_values = pauli_expectation_0)
        rho_standard_001 = standard_tomography(set_measure_basis = p_basis, expectation_values = pauli_expectation_001)
        rho_standard_005 = standard_tomography(set_measure_basis = p_basis, expectation_values = pauli_expectation_005)
        rho_standard_01  = standard_tomography(set_measure_basis = p_basis, expectation_values = pauli_expectation_01)
        rho_standard_02  = standard_tomography(set_measure_basis = p_basis, expectation_values = pauli_expectation_02)

        # Compute the fidelities
        fidelity_p_0    = fidelity(rho = rho_true, sigma = rho_standard_0, lib = "numpy")
        fidelity_p_001  = fidelity(rho = rho_true, sigma = rho_standard_001, lib = "numpy")
        fidelity_p_005  = fidelity(rho = rho_true, sigma = rho_standard_005, lib = "numpy")
        fidelity_p_01   = fidelity(rho = rho_true, sigma = rho_standard_01, lib = "numpy")
        fidelity_p_02   = fidelity(rho = rho_true, sigma = rho_standard_02, lib = "numpy")

        # Save the results in the list
        # For CS
        fidelities[0].append(CSfidelity_0)
        fidelities[1].append(CSfidelity_001)
        fidelities[2].append(CSfidelity_005)
        fidelities[3].append(CSfidelity_01)
        fidelities[4].append(CSfidelity_02)

        # For Standard tomography
        fidelities[5].append(fidelity_p_0)
        fidelities[6].append(fidelity_p_001)
        fidelities[7].append(fidelity_p_005)
        fidelities[8].append(fidelity_p_01)
        fidelities[9].append(fidelity_p_02)

    # Save the file with the results
    fidelities = np.array(fidelities)
    np.save(file = f"data_fidelities/data_fidelities_nexp{n_exp}", arr = fidelities)
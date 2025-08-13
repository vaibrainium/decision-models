## DDM
from typing import Callable, Dict, Optional

import numpy as np
import torch
from scipy import stats
from scipy.optimize import minimize


def get_psychometric_data(data, positive_direction="right"):
    """Extracts psychometric data and optionally fits a model."""
    unique_coh = np.unique(data["signed_coherence"])
    x_data = np.where(positive_direction == "left", -unique_coh, unique_coh)
    y_data = []
    trial_counts = []
    for coh in unique_coh:
        mask = (data["signed_coherence"] == coh) & (~np.isnan(data["choice"]))
        total_trials = np.sum(mask)

        if total_trials == 0:
            continue  # Skip coherence levels with no trials

        prop_positive = np.mean(data["choice"][mask] == (positive_direction == "right"))
        y_data.append(prop_positive)
        trial_counts.append(total_trials)

    # Convert to numpy arrays and sort
    x_data, y_data, trial_counts = map(np.array, zip(*sorted(zip(x_data, y_data, trial_counts, strict=False)), strict=False))

    return x_data, y_data


def get_chronometric_data(data, positive_direction="right"):
    """Computes reaction time statistics for different coherence levels."""
    unique_coh = np.unique(data["signed_coherence"])
    coherences, rt_median, rt_mean, rt_sd, rt_sem = [], [], [], [], []

    for coh in unique_coh:
        trials = data[data["signed_coherence"] == coh]
        if trials.empty:
            continue

        coherences.append(coh if positive_direction == "right" else -coh)
        rt_median.append(np.median(trials["rt"]))
        rt_mean.append(np.mean(trials["rt"]))
        rt_sd.append(np.std(trials["rt"]))
        rt_sem.append(stats.sem(trials["rt"], nan_policy="omit"))

    return map(np.array, zip(*sorted(zip(coherences, rt_median, rt_mean, rt_sd, rt_sem, strict=False)), strict=False))


class DriftDiffusionSimulator:
    def __init__(self, leak=True, time_dependence=True):
        """Initialize parameters for the diffusion model."""
        self.ndt = 0.1  # non-decision time
        self.a = 2  # boundary separation
        self.z = 0.5  # starting point (as a proportion of a)
        self.drift_gain = 7  # gain on the input to get drift rate
        self.drift_offset = 0.0
        self.variance = 1  # variance of the noise in the accumulation process
        self.dt = 0.001  # time step for simulation
        self.leak_rate = 0.01 if leak else 0  # leakage parameter
        self.time_constant = 1e-2 if time_dependence else 0  # urgency signal time constant

    def simulate_trials(self, stimulus):
        """Simulate trials of the diffusion model given a stimulus."""
        n_trials = stimulus.shape[0]
        n_timepoints = stimulus.shape[1]

        # Preallocate arrays for decision variables and responses
        evidence = np.full((n_trials, n_timepoints), np.nan)
        dv = np.full((n_trials, n_timepoints), np.nan)
        rt = np.full(n_trials, np.nan)
        choice = np.full(n_trials, np.nan)

        # Calculate drift rates based on stimulus and gain
        drift_rates = self.drift_gain * stimulus + self.drift_offset
        starting_point = self.z * self.a
        urgency = np.exp(np.arange(n_timepoints) * self.time_constant)
        noise = np.random.normal(0, np.sqrt(self.variance * self.dt), size=(n_trials, n_timepoints))
        for trial in range(n_trials):
            evidence[trial, 0] = starting_point
            dv[trial, 0] = starting_point
            for t in range(1, n_timepoints):
                if np.isnan(stimulus[trial, t]):
                    break  # no response

                momentary_evidence = drift_rates[trial, t - 1] * self.dt + noise[trial, t - 1]
                leakage = self.leak_rate * (evidence[trial, t - 1] - starting_point) * self.dt
                evidence[trial, t] = evidence[trial, t - 1] + momentary_evidence - leakage
                relative_evidence = evidence[trial, t] - starting_point
                dv[trial, t] = urgency[t] * (relative_evidence) + starting_point

                # Check for boundary crossing
                if dv[trial, t] >= self.a:
                    rt[trial] = t * self.dt + self.ndt
                    choice[trial] = 1
                    dv[trial, t:] = self.a  # Hold DV at boundary after crossing
                    break
                if dv[trial, t] <= 0:
                    rt[trial] = t * self.dt + self.ndt
                    choice[trial] = 0
                    dv[trial, t:] = 0  # Hold DV at boundary after crossing
                    break
        return rt, choice, dv


class DriftDiffusionSimulatorCUDA:
    def __init__(self, leak=True, time_dependence=True, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ndt = 0.1
        self.a = 2.0
        self.z = 0.5
        self.drift_gain = 7.0
        self.drift_offset = 0.0
        self.variance = 1.0
        self.dt = 0.001
        self.leak_rate = 0.01 if leak else 0.0
        self.time_constant = 1e-2 if time_dependence else 0.0

    def simulate_trials(self, stimulus):
        # Convert to tensor
        stimulus = torch.tensor(stimulus, device=self.device, dtype=torch.float32)
        n_trials, n_timepoints = stimulus.shape

        starting_point = self.z * self.a

        # Allocate arrays
        evidence = torch.full((n_trials, n_timepoints), float("nan"), device=self.device)
        dv = torch.full((n_trials, n_timepoints), float("nan"), device=self.device)
        rt = torch.full((n_trials,), float("nan"), device=self.device)
        choice = torch.full((n_trials,), float("nan"), device=self.device)
        done = torch.zeros(n_trials, dtype=torch.bool, device=self.device)

        # Drift rates and urgency
        drift_rates = self.drift_gain * stimulus + self.drift_offset
        urgency = torch.exp(torch.arange(n_timepoints, device=self.device) * self.time_constant)

        # Initialize
        evidence[:, 0] = starting_point
        dv[:, 0] = starting_point

        noise_std = torch.sqrt(torch.tensor(self.variance * self.dt, device=self.device))

        for t in range(1, n_timepoints):
            running = ~done
            if running.sum() == 0:
                break

            # Identify NaN stimulus (no response)
            nan_mask = torch.isnan(stimulus[running, t])
            if nan_mask.any():
                done[running.nonzero(as_tuple=True)[0][nan_mask]] = True
                continue

            prev_ev = evidence[running, t - 1]

            momentary_evidence = drift_rates[running, t - 1] * self.dt + torch.randn(running.sum(), device=self.device) * noise_std
            leakage = self.leak_rate * (prev_ev - starting_point) * self.dt

            ev_new = prev_ev + momentary_evidence - leakage
            evidence[running, t] = ev_new

            relative_ev = ev_new - starting_point
            dv_new = urgency[t] * relative_ev + starting_point
            dv[running, t] = dv_new

            # Boundary detection
            crossed_up = dv_new >= self.a
            crossed_down = dv_new <= 0
            crossed = crossed_up | crossed_down

            if crossed.any():
                just_finished_idx = running.nonzero(as_tuple=True)[0][crossed]

                rt[just_finished_idx] = t * self.dt + self.ndt
                choice[just_finished_idx] = torch.where(crossed_up[crossed], 1.0, 0.0)

                # Fill DV after boundary crossing
                a_fill = torch.full((just_finished_idx.shape[0], n_timepoints - t), self.a, device=self.device)
                zero_fill = torch.zeros((just_finished_idx.shape[0], n_timepoints - t), device=self.device)
                up_mask = crossed_up[crossed].unsqueeze(1)

                dv[just_finished_idx, t:] = torch.where(up_mask, a_fill, zero_fill)

                done[just_finished_idx] = True

        return rt.cpu().numpy(), choice.cpu().numpy(), dv.cpu().numpy()


class LikelihoodCalculator:
    def __init__(self, nbins=9):
        """
        Class to compute Ratcliff-style QMLE likelihoods for diffusion model fits.

        Args:
            nbins (int): Number of quantile bins for RT likelihood estimation (default=9).

        """
        self.nbins = nbins
        self.eps = 1e-24  # small number to avoid log(0)
        self.rt_nllh_weight = 1  #  1000.0

    def calculate_llh_QMLE(self, rt_model, rt_data):
        """
        Quasi-Maximum Likelihood (Ratcliff quantile likelihood) for RT distributions.

        Args:
            rt_model (array-like): simulated RTs (ms)
            rt_data (array-like): empirical RTs (ms)

        Returns:
            nllh (float): negative log likelihood for RT data given model

        """
        rt_model = np.asarray(rt_model)
        rt_data = np.sort(np.asarray(rt_data))
        n = len(rt_data)
        if self.nbins > n:
            # Fallback: Use fewer bins or skip likelihood
            nbins_used = max(1, n // 2)  # or just 1 bin if very low data
        else:
            nbins_used = self.nbins

        quantiles = np.linspace(0, 1, nbins_used + 1)
        bin_edges = np.quantile(rt_data, quantiles)

        # Expand edges to fully cover model RTs
        bin_edges[0] = min(bin_edges[0], np.min(rt_model))
        bin_edges[-1] = max(bin_edges[-1], np.max(rt_model))

        counts_per_bin = np.histogram(rt_data, bins=bin_edges)[0]
        probs_per_bin = np.histogram(rt_model, bins=bin_edges)[0] / len(rt_model)
        # for data_count, model_prob in zip(counts_per_bin, probs_per_bin):
        #     if data_count > 0 and model_prob == 0:
        #         probs_per_bin += self.eps
        nllh = -np.sum(counts_per_bin * np.log(probs_per_bin + self.eps))
        return nllh

    def calculate_choice_likelihood(self, choice_model, choice_data):
        """
        Negative log likelihood for choice proportions.

        Args:
            choice_model (array-like): simulated choices
            choice_data (array-like): empirical choices

        Returns:
            nllh (float): negative log likelihood for choice data given model

        """
        choice_model = np.asarray(choice_model)
        choice_data = np.asarray(choice_data)

        categories = np.unique(choice_data)
        nllh = 0.0
        for cat in categories:
            n_cat_data = np.sum(choice_data == cat)
            p_cat_model = np.mean(choice_model == cat)
            nllh -= n_cat_data * np.log(p_cat_model + self.eps)
        return nllh

    def calculate_total_llh_per_choice(self, prediction, data):
        """
        Computes Ratcliff-style QMLE negative log likelihood summed over coherence × choice subsets.

        Args:
            prediction (dict): keys 'signed_coherence', 'choice', 'rt' with arrays of model predictions
            data (dict): keys 'signed_coherence', 'choice', 'rt' with arrays of empirical data

        Returns:
            total_nllh (float): total negative log likelihood

        """
        # remove trials with NaN choices or RTs
        mask_data = (~np.isnan(data["choice"])) & (~np.isnan(data["rt"]))
        data = {k: v[mask_data] for k, v in data.items()}
        mask_model = (~np.isnan(prediction["choice"])) & (~np.isnan(prediction["rt"]))
        prediction = {k: v[mask_model] for k, v in prediction.items()}

        total_nllh = 0.0
        unique_cohs = np.unique(data["signed_coherence"])
        unique_choices = np.unique(data["choice"])

        for coh in unique_cohs:
            mask_data_coh = data["signed_coherence"] == coh
            mask_model_coh = prediction["signed_coherence"] == coh

            if np.sum(mask_data_coh) == 0 or np.sum(mask_model_coh) == 0:
                continue

            choice_data_coh = data["choice"][mask_data_coh]
            choice_model_coh = prediction["choice"][mask_model_coh]

            if len(choice_data_coh) == 0 or len(choice_model_coh) == 0:
                continue

            # Choice likelihood per coherence
            nllh_choice = 0.0
            for choice_val in unique_choices:
                n_choice_data = np.sum(choice_data_coh == choice_val)
                p_choice_model = np.mean(choice_model_coh == choice_val)
                nllh_choice -= n_choice_data * np.log(p_choice_model + self.eps)
            total_nllh += nllh_choice

            # RT likelihood per choice
            for choice_val in unique_choices:
                mask_data = mask_data_coh & (data["choice"] == choice_val)
                mask_model = mask_model_coh & (prediction["choice"] == choice_val)

                rt_data_sub = data["rt"][mask_data]
                rt_model_sub = prediction["rt"][mask_model]

                if len(rt_data_sub) == 0 or len(rt_model_sub) == 0:
                    continue

                nllh_rt = self.calculate_llh_QMLE(rt_model_sub, rt_data_sub)
                total_nllh += nllh_rt * self.rt_nllh_weight

        return total_nllh


class DecisionModel:
    """Main class for decision-making model simulation and fitting."""

    def __init__(
        self,
        model_name: str = "DDM",
        enable_leak: bool = True,
        enable_time_dependence: bool = True,
        likelihood_function: Optional[Callable] = None,
        likelihood_params: Optional[Dict] = None,
        device=None,
    ):
        """
        Initialize the decision model with parameters.

        Args:
            model_name (str): Model variant name, e.g., 'DDM', 'Race', 'LCA'.
            enable_leak (bool): Whether to include leak in the accumulation process.
            enable_time_varying_drift (bool): Whether to include time-dependent drift (urgency).
            likelihood_function (Callable or None): Custom likelihood function. Defaults to internal QMLE.
            likelihood_params (dict or None): Parameters to configure the likelihood calculation (e.g., nbins).

        """
        self.model_name = model_name
        self.enable_leak = enable_leak
        self.enable_time_dependence = enable_time_dependence

        # Initialize simulator based on model_name — extend this as needed
        if model_name.upper() == "DDM":
            if device == "cuda":
                self.simulator = DriftDiffusionSimulatorCUDA(leak=enable_leak, time_dependence=enable_time_dependence, device=device)
            else:
                self.simulator = DriftDiffusionSimulator(leak=enable_leak, time_dependence=enable_time_dependence)
        else:
            raise NotImplementedError(f"Simulator for model '{model_name}' not implemented.")

        # Setup likelihood calculator with default or user-defined function
        if likelihood_params is None:
            likelihood_params = {"nbins": 9}

        if likelihood_function is not None:
            self.likelihood_calculator = likelihood_function
        else:
            self.likelihood_calculator = LikelihoodCalculator(**likelihood_params)

    def fit(self, data, stimulus, optimization_method=None, optimizer_options=None, n_reps=1, seed=42):
        """
        Fit the model to empirical data using optimization.

        Args:
            data (DataFrame): columns 'signed_coherence', 'choice', 'rt', 'prior_block'.
            stimulus (ndarray): shape (n_trials, n_timepoints)
            optimizer_options (dict): options for optimizer.
            n_reps (int): number of replicate simulations per evaluation to reduce noise.
            seed (int): base RNG seed for deterministic likelihoods.
            rt_weight (float): weight for RT likelihood.
            choice_weight (float): weight for choice likelihood.

        """
        if optimizer_options is None:
            optimizer_options = {"maxiter": 200, "disp": True}

        # --- Parameter setup ---
        self.all_params = {
            # common parameters
            "ndt": (0.1, (0.05, 0.3)),
            "drift_gain": (7.0, (1.0, 20.0)),
            "variance": (1.0, (0.1, 5.0)),
            "a_1": (2.0, (0.5, 5.0)),
            "z_1": (0.5, (0.1, 0.9)),
            "drift_offset_1": (0.0, (-5.0, 5.0)),
            "a_2": (2.0, (0.5, 5.0)),
            "z_2": (0.5, (0.1, 0.9)),
            "drift_offset_2": (0.0, (-5.0, 5.0)),
        }
        if self.enable_leak:
            self.all_params["leak_rate"] = (0.1, (0.0, 1.0))
        if self.enable_time_dependence:
            self.all_params["time_constant"] = (1e-2, (0.0, 0.1))

        param_names = list(self.all_params.keys())
        initial_params = [self.all_params[n][0] for n in param_names]
        bounds = [self.all_params[n][1] for n in param_names]

        # --- Objective function ---
        def objective_function(params, lambda_l1=0.01):
            total_nllh = 0.0
            for idx_prior, prior in enumerate(["equal", "unequal"]):
                prior_mask = data["prior_block"] == prior
                # Map params into simulator
                for name, value in zip(param_names, params):
                    if (name.endswith("_1") and idx_prior == 0) or (name.endswith("_2") and idx_prior == 1):
                        base_name = name[:-2]
                        setattr(self.simulator, base_name, value)
                    else:
                        setattr(self.simulator, name, value)

                # Accumulate across replicates to reduce noise
                # all_preds = []
                # for rep in range(n_reps):
                np.random.seed(seed)
                rt_model, choice_model, _ = self.simulator.simulate_trials(stimulus[prior_mask])
                # all_preds.append((rt_model, choice_model))

                # Average predictions over replicates
                # rt_model = np.mean([p[0] for p in all_preds], axis=0)
                # choice_model = np.round(np.mean([p[1] for p in all_preds], axis=0))

                prediction = {
                    "signed_coherence": stimulus[prior_mask, 0],
                    "choice": choice_model,
                    "rt": rt_model,
                }

                # Calculate weighted NLLH
                total_nllh += self.likelihood_calculator.calculate_total_llh_per_choice(prediction, data[prior_mask])

            if not np.isfinite(total_nllh):
                return 1e10

            # L1 penalty if desired
            total_nllh += lambda_l1 * np.sum(np.abs(params))
            return total_nllh

        # --- Stage 1: coarse Powell search ---
        stage1 = minimize(objective_function, x0=initial_params, bounds=bounds, method="Powell", options={"maxiter": 100, "disp": True})

        # --- Stage 2: refine with L-BFGS-B ---
        stage2 = minimize(objective_function, x0=stage1.x, bounds=bounds, method="L-BFGS-B", options=optimizer_options)

        return stage2

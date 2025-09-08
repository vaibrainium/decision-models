import torch
from torch import nn


class SoftBoundDDM_RT(nn.Module):
    def __init__(self, leak=True, time_dependence=True, beta=50.0):
        super().__init__()
        # Learnable parameters
        self.ndt = nn.Parameter(torch.tensor(0.1))
        self.a = nn.Parameter(torch.tensor(2.0))
        self.z = nn.Parameter(torch.tensor(0.5))
        self.drift_gain = nn.Parameter(torch.tensor(7.0))
        self.drift_offset = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(beta))

        # Fixed parameters
        self.variance = 1.0
        self.dt = 0.001
        self.leak_rate = 0.01 if leak else 0.0
        self.time_constant = 1e-2 if time_dependence else 0.0

    def forward(self, stimulus):
        n_trials, n_timepoints = stimulus.shape
        device = stimulus.device

        evidence = torch.zeros((n_trials, n_timepoints), device=device)
        dv = torch.zeros_like(evidence)

        starting_point = self.z * self.a
        urgency = torch.arange(n_timepoints, device=device) * self.time_constant
        drift_rates = self.drift_gain * stimulus + self.drift_offset
        noise = torch.randn_like(stimulus) * torch.sqrt(torch.tensor(self.variance * self.dt, device=device))

        evidence[:, 0] = starting_point
        dv[:, 0] = starting_point

        for t in range(1, n_timepoints):
            momentary_evidence = drift_rates[:, t - 1] * self.dt + noise[:, t - 1]
            leakage = self.leak_rate * (evidence[:, t - 1] - starting_point) * self.dt
            evidence[:, t] = evidence[:, t - 1] + momentary_evidence - leakage
            dv[:, t] = urgency[t] * (evidence[:, t] - starting_point) + starting_point

        # Compute soft decision hazards
        h1 = torch.sigmoid(self.beta * (dv - self.a))  # hazard of choice 1
        h0 = torch.sigmoid(self.beta * (0.0 - dv))  # hazard of choice 0
        return dv, h1, h0

    def compute_likelihood(self, stimulus, observed_choice, observed_rt):
        """
        stimulus: (n_trials, n_timepoints)
        observed_choice: (n_trials,) 0 or 1
        observed_rt: (n_trials,) in seconds
        """
        n_trials, n_timepoints = stimulus.shape
        device = stimulus.device
        dt = self.dt

        _, h1, h0 = self.forward(stimulus)

        # Convert observed RT to time index
        rt_idx = torch.clamp((observed_rt / dt).long(), max=n_timepoints - 1)

        # Compute survival probability
        survival = torch.cumprod(1 - h1 - h0 + 1e-8, dim=1)
        survival_shifted = torch.cat([torch.ones((n_trials, 1), device=device), survival[:, :-1]], dim=1)

        # Likelihood at decision time
        prob1 = h1 * survival_shifted
        prob0 = h0 * survival_shifted

        # Select observed choice
        idx = torch.arange(n_trials, device=device)
        choice_prob = torch.where(observed_choice == 1, prob1[idx, rt_idx], prob0[idx, rt_idx])

        # Negative log-likelihood
        nll = -torch.sum(torch.log(choice_prob + 1e-8))
        return nll

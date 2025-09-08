import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim


class DifferentiableDDMTrainer(nn.Module):
    def __init__(self, dt=0.001, max_t=2.0, sigma=1.0, n_rt_bins=10, lr=1e-2, device="cpu"):
        super().__init__()
        self.dt = dt
        self.max_t = max_t
        self.sigma = sigma
        self.n_rt_bins = n_rt_bins
        self.device = device

        # Learnable DDM parameters
        self.params = nn.ParameterDict(
            {
                "a": nn.Parameter(torch.tensor(1.0)),  # boundary separation
                "z": nn.Parameter(torch.tensor(0.0)),  # starting bias (fraction of boundary)
                "ndt": nn.Parameter(torch.tensor(0.3)),  # non-decision time
                "drift_gain": nn.Parameter(torch.tensor(1.0)),  # drift scaling
            },
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, params=None):
        if params is None:
            params = {k: v for k, v in self.params.items()}

        n_trials = x.shape[0]
        dt = self.dt
        max_t = self.max_t
        steps = int(max_t / dt)

        a = params["a"]
        z = params["z"]
        ndt = params["ndt"]
        drift_gain = params["drift_gain"]

        dv = torch.zeros(n_trials, device=self.device) + (z * a)
        pred_rt = torch.zeros(n_trials, device=self.device)
        pred_choice = torch.zeros(n_trials, device=self.device)
        active = torch.ones(n_trials, dtype=torch.bool, device=self.device)

        up_b = a
        low_b = -a
        drift = drift_gain * x.squeeze()

        for t in range(steps):
            if not active.any():
                break

            noise = torch.randn_like(dv[active]) * (self.sigma * torch.sqrt(torch.tensor(dt)))
            dv[active] += drift[active] * dt + noise

            hit_upper = dv[active] >= (up_b - 1e-6)
            hit_lower = dv[active] <= (low_b + 1e-6)
            just_hit = hit_upper | hit_lower

            if just_hit.any():
                indices = active.nonzero().squeeze()[just_hit]
                pred_rt[indices] = t * dt + ndt
                pred_choice[indices] = hit_upper[just_hit].float()

            active_indices = active.nonzero().squeeze()
            active[active_indices[just_hit]] = False

        # Fill timeouts
        pred_rt[active] = max_t + ndt
        pred_choice[active] = 0.5

        return pred_rt, pred_choice

    def compute_loss(self, pred_rt, pred_choice, true_rt, true_choice):
        # Choice loss
        choice_loss = F.binary_cross_entropy(pred_choice, true_choice.float())

        # RT histogram KL divergence
        rt_bins = torch.linspace(true_rt.min(), true_rt.max(), self.n_rt_bins + 1, device=self.device)
        true_hist = torch.histc(true_rt, bins=self.n_rt_bins, min=rt_bins[0], max=rt_bins[-1])
        pred_hist = torch.histc(pred_rt, bins=self.n_rt_bins, min=rt_bins[0], max=rt_bins[-1])

        true_hist = true_hist / true_hist.sum()
        pred_hist = pred_hist / pred_hist.sum()

        rt_loss = F.kl_div(pred_hist.log(), true_hist, reduction="batchmean")

        return choice_loss + rt_loss

    def fit(self, df, stim, steps=100, clip=5.0):
        x = torch.tensor(stim, device=self.device).float().view(-1, 1)
        true_rt = torch.tensor(df["rt"].values, device=self.device).float()
        true_choice = torch.tensor(df["choice"].values, device=self.device).float()

        for step in range(steps):
            self.optimizer.zero_grad()
            pred_rt, pred_choice = self.forward(x)
            loss = self.compute_loss(pred_rt, pred_choice, true_rt, true_choice)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), clip)
            self.optimizer.step()

            if step % 10 == 0 or step == steps - 1:
                print(f"Step {step}: Loss = {loss.item():.4f}")

        learned_params = {k: v.detach().cpu().item() for k, v in self.params.items()}
        return learned_params

    def plot_diagnostics(self, df, stim):
        with torch.no_grad():
            x = torch.tensor(stim, device=self.device).float().view(-1, 1)
            pred_rt, pred_choice = self.forward(x)
            pred_rt = pred_rt.cpu().numpy()
            pred_choice = pred_choice.cpu().numpy()

        # Psychometric curve
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        for sign, label in zip([1, 0], ["Choice 1", "Choice 0"]):
            mask = df["choice"].values == sign
            plt.scatter(stim[mask], df["choice"].values[mask], alpha=0.3, label=f"True {label}")
        plt.scatter(stim, pred_choice, alpha=0.3, color="red", label="Predicted")
        plt.xlabel("Stimulus")
        plt.ylabel("Choice Probability")
        plt.title("Psychometric Curve")
        plt.legend()

        # Chronometric curve
        plt.subplot(1, 2, 2)
        plt.scatter(stim, df["rt"].values, alpha=0.3, label="True")
        plt.scatter(stim, pred_rt, alpha=0.3, color="red", label="Predicted")
        plt.xlabel("Stimulus")
        plt.ylabel("RT (s)")
        plt.title("Chronometric Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd

    # Fake data
    n_trials = 500
    stim = np.random.choice([-0.2, -0.1, 0.0, 0.1, 0.2], size=n_trials)
    true_rt = np.random.normal(0.6, 0.1, size=n_trials)
    true_choice = (stim + np.random.normal(0, 0.1, size=n_trials)) > 0
    df = pd.DataFrame({"rt": true_rt, "choice": true_choice.astype(int)})

    trainer = DifferentiableDDMTrainer(device="cpu")
    learned = trainer.fit(df, stim, steps=100)
    print("Learned parameters:", learned)

    trainer.plot_diagnostics(df, stim)

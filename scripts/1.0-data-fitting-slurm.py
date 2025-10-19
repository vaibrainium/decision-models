import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import dir_config
from src.ddm_independent_parameter_models import *

# config
compiled_dir = Path(dir_config.data.compiled)
processed_dir = Path(dir_config.data.processed)
ddm_dir = Path(dir_config.data.processed, "ddm")


ENABLE_LEAK = True
ENABLE_TIME_DEPENDENCY = True
model_folder = f"ddm_leak_{ENABLE_LEAK}_urgency_{ENABLE_TIME_DEPENDENCY}"
output_dir = Path(ddm_dir, model_folder)
output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # get session_id from command line arguments
    parser = argparse.ArgumentParser(description="Fit session data")
    parser.add_argument("--session_id", type=int, required=True, help="ID of the session to fit")
    args = parser.parse_args()
    idx_session = args.session_id

    # load behavior data
    behavior_df = pd.read_csv(Path(ddm_dir, "behavior_data.csv"))
    session_ids = behavior_df["session_id"].unique()
    session_id = session_ids[idx_session]

    data = behavior_df[behavior_df["session_id"] == session_id]
    data = data[["signed_coherence", "choice", "rt", "prior_block"]].reset_index(drop=True)
    data["choice"] = data["choice"].astype(int)
    # Get stimulus length
    stimulus_length = int(np.max(data["rt"]) * 1000)
    stimulus = np.tile(data["signed_coherence"].to_numpy().reshape(-1, 1), (1, stimulus_length)) / 100

    # Create model
    model = DecisionModel(enable_leak=ENABLE_LEAK, enable_time_dependence=ENABLE_TIME_DEPENDENCY, device="cuda")

    optimizer_options = {"maxiter": 1000, "maxls": 50, "ftol": 1e-6, "gtol": 1e-4, "disp": False}

    result = model.fit(
        data=data,
        stimulus=stimulus,
        max_iterations=optimizer_options["maxiter"],
        n_reps=5,  # Reduce for faster fitting
        seed=42,
        l1_weight=0.01,
        verbose=True,
    )

    # Save results
    with open(Path(output_dir, f"{session_id}.pkl"), "wb") as f:
        pickle.dump({"models": model, "results": result, "session_id": session_id}, f)

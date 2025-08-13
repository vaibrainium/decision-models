from pathlib import Path
import json
from omegaconf import OmegaConf


# Load the YAML configuration file in config directory
def load_config(filename):
	return OmegaConf.load(Path(__file__).parent / filename)


def load_json(filename):
	with open(Path(__file__).parent / filename, "r") as file:
		return json.load(file)


# Example of accessing a specific configuration file
dir_config = load_config("dir-config.yaml")

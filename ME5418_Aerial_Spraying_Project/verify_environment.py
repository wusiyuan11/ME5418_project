"""Verify the project interpreter, dependencies, and Gymnasium API."""

# Import the interpreter module to report the exact Python version.
import sys
# Import Gymnasium and its official environment checker.
import gymnasium
from gymnasium.utils.env_checker import check_env
# Import Matplotlib and NumPy to verify required scientific packages.
import matplotlib
import numpy
# Import the local environment that will be validated.
from spraying_env import SprayingEnv

# Print exact versions so the software environment is reproducible.
print(f"Python: {sys.version.split()[0]}")
print(f"NumPy: {numpy.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Gymnasium: {gymnasium.__version__}")
# Construct the project environment with deterministic settings.
environment = SprayingEnv(seed=7)
# Run Gymnasium's official structural and behavioral checks.
check_env(environment, skip_render_check=True)
# Close resources after successful validation.
environment.close()
# Print an unambiguous success message only after every check passes.
print("Environment verification: PASS")

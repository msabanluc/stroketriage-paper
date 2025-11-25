import os
import random as rn
import numpy as np
import tensorflow as tf

def enforce_reproducibility(seed=2023):
    # Check PYTHONHASHSEED
    current_seed = os.environ.get("PYTHONHASHSEED")
    if current_seed != "0":
        raise EnvironmentError(
            "PYTHONHASHSEED must be set to '0' for full reproducibility. "
            "Set this in your shell or conda environment before running the script."
        )

    # Set seeds for Python, NumPy, and TensorFlow
    rn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

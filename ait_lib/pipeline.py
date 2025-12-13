import pandas as pd
import numpy as np
from scipy.stats import entropy

class SignalProcessor:
    """
    Handles the transformation from Raw Price -> MILS Grid.
    """
    def __init__(self, window_size=50):
        self.window_size = window_size

    def get_acceleration(self, series):
        """
        Calculates Causal Force (F=ma). 
        For non-stationary assets, uses 2nd derivative (Acceleration).
        """
        velocity = series.diff()
        acceleration = velocity.diff()
        return acceleration.dropna()

    def mils_encoding(self, signal, n_bins=4):
        """
        Discretizes signal into 4-bit Quantile Grid.
        Preserves ordinal structure while filtering micro-noise.
        """
        try:
            bins = pd.qcut(signal, n_bins, labels=False, duplicates='drop')
            # One-hot encode
            grid = np.eye(n_bins)[bins.astype(int)]
            return grid
        except ValueError:
            # Fallback for flat signals
            return np.zeros((len(signal), n_bins))

class Diagnostics:
    """
    The Seismograph tools.
    """
    @staticmethod
    def calculate_entropy(probs):
        """
        Calculates Shannon Entropy of the rule distribution.
        H(t) = -sum(p * log(p))
        """
        return entropy(probs, base=2)

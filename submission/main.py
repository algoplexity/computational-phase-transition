import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer

# ==============================================================================
# ALGOPLEXITY SUBMISSION: QUANTIZED CYBERNETIC FUSION
# ==============================================================================

# --- 1. ARCHITECTURE (The Physicist) ---
class TinyRecursiveModel(nn.Module):
    def __init__(self, input_width=4, hidden_dim=64, num_classes=9):
        super(TinyRecursiveModel, self).__init__()
        self.encoder = nn.Linear(input_width, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        _, h_n = self.rnn(encoded)
        return self.head(h_n.squeeze(0))

# --- 2. ARCHITECTURE (The Governor) ---
class QuantizedCalibrator:
    """
    Non-Linear Governor: Bins the evidence to handle high-energy outliers.
    """
    def __init__(self, n_bins=15):
        self.quantizer = KBinsDiscretizer(n_bins=n_bins, encode='onehot', strategy='uniform')
        self.thermostat = LogisticRegression(penalty=None, solver='lbfgs')

    def fit(self, evidence_vectors, labels):
        # 1. Bin the continuous evidence (Bits) into discrete Energy Levels
        evidence_binned = self.quantizer.fit_transform(evidence_vectors)
        # 2. Learn the probability weight for each Level
        self.thermostat.fit(evidence_binned, labels)
        
    def predict_proba(self, evidence_vector):
        vec = np.array(evidence_vector)
        if vec.ndim == 1: vec = vec.reshape(1, -1)
        # Transform raw bits -> Energy Bins -> Probability
        vec_binned = self.quantizer.transform(vec)
        return self.thermostat.predict_proba(vec_binned)[:, 1][0]

# ==============================================================================
# 3. SENSORY PROCESSING PIPELINE
# ==============================================================================

def get_acceleration(series):
    """Calculates Velocity (First Derivative) as the input signal."""
    return series.diff().dropna()

def quantile_encode(acc, width=4):
    """Canonical 4-Bin Encoding. Robust to length mismatches."""
    try:
        if len(acc) < width: return None
        bins = pd.qcut(acc, q=width, labels=False, duplicates='drop')
        binary_grid = np.eye(width)[bins.astype(int)]
        return binary_grid
    except:
        return None

def get_entropy_signal(acc, physicist, device, window=30):
    """Calculates the Entropic Ambiguity (Confusion) of the Physicist."""
    binary = quantile_encode(acc)
    if binary is None or len(binary) < window: return None

    indices = range(len(binary) - window)
    if not indices: return None
    
    # Create batch of sliding windows
    X_windows = np.array([binary[i : i+window] for i in indices])
    X_tensor = torch.FloatTensor(X_windows).to(device)

    with torch.no_grad():
        logits = physicist(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    # Shannon Entropy H(t)
    entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    
    # Pad to match original length
    full_entropy = np.full(len(acc), np.nan)
    full_entropy[window:] = entropy
    return full_entropy

def mdl_cost(signal):
    """Minimum Description Length (Energy Cost)."""
    clean_sig = signal[~np.isnan(signal)]
    n = len(clean_sig)
    if n < 2: return 0
    var = np.var(clean_sig)
    if var < 1e-9: var = 1e-9
    nll = 0.5 * n * np.log(2 * np.pi * var) + 0.5 * n
    return np.log(n) + nll

def calculate_evidence_gap(signal, break_idx):
    """Calculates Information Gain (Bits) of a Structural Break hypothesis."""
    if break_idx < 10 or break_idx > len(signal) - 10: return 0.0
    pre = signal[:break_idx]
    post = signal[break_idx:]
    if len(pre) < 10 or len(post) < 10: return 0.0
    
    cost_unified = mdl_cost(signal)
    cost_split = mdl_cost(pre) + mdl_cost(post)
    return (cost_unified - cost_split) / np.log(2)

def extract_cybernetic_vector(dataset, physicist, device):
    """Extracts [Logic_Bits, Energy_Bits] for a single asset."""
    df = dataset.sort_values('time')
    series = df['value']
    
    # CrunchDAO specific: Break is at the start of period 0 (Test Segment)
    break_idx = len(df[df['period'] != 0])
    if break_idx == 0: break_idx = len(df) // 2 
    
    acc = get_acceleration(series)
    if len(acc) < 30: return [0.0, 0.0]
    adj_break = break_idx - 1 
    
    # Channel A: Logic (Entropy)
    entropy = get_entropy_signal(acc, physicist, device)
    if entropy is not None:
        logic_bits = calculate_evidence_gap(entropy, adj_break)
    else:
        logic_bits = 0.0
        
    # Channel B: Energy (Magnitude)
    energy = np.abs(acc.values)
    energy_bits = calculate_evidence_gap(energy, adj_break)
    
    return [logic_bits, energy_bits]

# ==============================================================================
# 4. PLATFORM INTERFACE (Train / Infer)
# ==============================================================================

def train(X_train: pd.DataFrame, y_train: pd.Series, model_directory_path: str):
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu") # CPU is sufficient for inference loop
    
    print("Algoplexity: Initializing Quantized Cybernetics...")
    
    # 1. Load the Pre-Trained Physicist (Universal Prior)
    physicist = TinyRecursiveModel()
    
    # Robust path checking for the uploaded model file
    weights_path = 'trm_expert.pth'
    if not os.path.exists(weights_path):
        # Fallback if platform places it in resources
        weights_path = os.path.join('resources', 'trm_expert.pth')
        
    if os.path.exists(weights_path):
        physicist.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"✅ Loaded Universal Prior: {weights_path}")
    else:
        print("⚠️ CRITICAL: trm_expert.pth not found. Model will use random weights!")
        
    physicist.eval()
    
    # 2. Extract Features
    all_ids = X_train.index.unique(level='id').values
    valid_ids = [i for i in all_ids if i in y_train.index]
    
    # OPTIMIZATION: Subsample to 1000 assets to prevent Time Limit Exceeded
    # The Governor converges quickly; it doesn't need 5000+ samples.
    if len(valid_ids) > 1000:
        print(f"Subsampling 1000 assets from {len(valid_ids)} for speed...")
        sampled_ids = np.random.choice(valid_ids, size=1000, replace=False)
    else:
        sampled_ids = valid_ids
    
    print(f"Extracting features for {len(sampled_ids)} assets...")
    vectors = []
    labels = []
    
    for uid in sampled_ids:
        group = X_train.loc[uid]
        vec = extract_cybernetic_vector(group, physicist, device)
        vectors.append(vec)
        labels.append(y_train.loc[uid])
            
    X_meta = np.array(vectors)
    y_meta = np.array(labels)
    
    # 3. Train the Quantized Governor
    print("Calibrating Governor...")
    governor = QuantizedCalibrator(n_bins=15)
    governor.fit(X_meta, y_meta)
    
    # 4. Save Artifacts for Inference Phase
    # We must save the Governor (it contains the learned calibration)
    joblib.dump(governor, os.path.join(model_directory_path, 'governor.joblib'))
    # We save the physicist configuration just in case
    torch.save(physicist.state_dict(), os.path.join(model_directory_path, 'physicist.pth'))
    print("Artifacts saved.")

def infer(X_test, model_directory_path: str):
    device = torch.device("cpu")
    
    # Load Physicist
    physicist = TinyRecursiveModel()
    physicist.load_state_dict(torch.load(os.path.join(model_directory_path, 'physicist.pth'), map_location=device))
    physicist.eval()
    
    # Load Governor
    governor = joblib.load(os.path.join(model_directory_path, 'governor.joblib'))
    
    # Signal readiness
    yield 
    
    for dataset in X_test:
        vec = extract_cybernetic_vector(dataset, physicist, device)
        prob = governor.predict_proba(vec)
        yield prob

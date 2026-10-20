"""
Easy-to-use loader for the trained and frozen Physics-Informed DeepONet weights and biases.
"""
import os
import json
import numpy as np

WEIGHTS_DIR = os.path.dirname(os.path.abspath(__file__))

def get_metadata():
    """Load model architecture metadata and configuration."""
    with open(os.path.join(WEIGHTS_DIR, 'model_metadata.json'), 'r') as f:
        return json.load(f)

def load_numpy_weights(model_type='inner'):
    """
    Load weights and biases as a dictionary of NumPy arrays.
    model_type: 'inflow', 'inner', or 'outflow'
    """
    path = os.path.join(WEIGHTS_DIR, f'{model_type}_weights_biases.npz')
    data = np.load(path)
    return {k: data[k] for k in data.files}

def load_jax_params(model_type='inner'):
    """
    Load original JAX parameter tuple (branch_params, trunk_params).
    model_type: 'inflow', 'inner', or 'outflow'
    """
    import pickle
    path = os.path.join(WEIGHTS_DIR, f'{model_type}_edge_frozen_params.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_torch_weights(model_type='inner'):
    """
    Load weights and biases as PyTorch tensors.
    model_type: 'inflow', 'inner', or 'outflow'
    """
    import torch
    path = os.path.join(WEIGHTS_DIR, f'{model_type}_weights_biases.pt')
    return torch.load(path, weights_only=True)

def load_pretrained_model(model_type='inner'):
    """
    Load fully assembled PyTorch PretrainedDeepONet model initialized with weights.
    model_type: 'inflow', 'inner', or 'outflow'
    """
    import sys
    engine_dir = os.path.join(os.path.dirname(WEIGHTS_DIR), 'Engine')
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    from QGPINNs_Engine import PretrainedDeepONet
    return PretrainedDeepONet(model_type=model_type, weights_dir=WEIGHTS_DIR)

if __name__ == '__main__':
    meta = get_metadata()
    print("=== DeepONet Model Metadata ===")
    print(json.dumps(meta, indent=2))

    for m_type in ['inflow', 'inner', 'outflow']:
        w = load_numpy_weights(m_type)
        print(f"\nSuccessfully loaded {m_type.upper()} model:")
        print(f"  Total weight/bias tensors: {len(w)}")
        print(f"  Sample tensor shape (branch layer 0): {w['branch_layer_0_weight'].shape}")
        print(f"  Sample tensor shape (trunk layer 0):  {w['trunk_layer_0_weight'].shape}")

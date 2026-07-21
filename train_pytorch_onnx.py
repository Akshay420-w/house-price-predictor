import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. DATA PREPROCESSING & GENERATION
# ==========================================
# Features: [GrLivArea, OverallQual, TotalBsmtSF, GarageCars, FullBath]
def get_dataset():
    n_samples = 1500
    gr_liv_area = np.random.normal(1500, 500, n_samples).clip(300, 4000)
    overall_qual = np.random.randint(1, 11, n_samples)
    total_bsmt_sf = np.random.normal(1000, 400, n_samples).clip(0, 3000)
    garage_cars = np.random.randint(0, 5, n_samples)
    full_bath = np.random.randint(1, 4, n_samples)

    # Base pricing logic + realistic noise
    target_price = (
        (gr_liv_area * 80) +
        (overall_qual * 15000) +
        (total_bsmt_sf * 50) +
        (garage_cars * 10000) +
        (full_bath * 12000) +
        20000 +
        np.random.normal(0, 10000, n_samples)
    )

    X = np.column_stack([gr_liv_area, overall_qual, total_bsmt_sf, garage_cars, full_bath])
    y = target_price.reshape(-1, 1)
    
    return X, y

X_raw, y_raw = get_dataset()

# Calculate Standardization Parameters (Pure NumPy to avoid sklearn/scipy memory issues)
mean_X = X_raw.mean(axis=0)
std_X = X_raw.std(axis=0)
std_X[std_X == 0] = 1.0  # Prevent division by zero

mean_y = float(y_raw.mean())
std_y = float(y_raw.std())

X_scaled = (X_raw - mean_X) / std_X
y_scaled = (y_raw - mean_y) / std_y

# Save parameters to scaler.json for web client scaling
scaler_params = {
    "mean_X": mean_X.tolist(),
    "std_X": std_X.tolist(),
    "mean_y": mean_y,
    "std_y": std_y
}

with open("scaler.json", "w") as f:
    json.dump(scaler_params, f, indent=4)

print("Saved scaling parameters to scaler.json")

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================
class MonotonicHousePredictor(nn.Module):
    def __init__(self, input_dim):
        super(MonotonicHousePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Softplus(),  # Smooth activation allowing continuous derivatives
            nn.Linear(64, 32),
            nn.Softplus(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

model = MonotonicHousePredictor(input_dim=5)

# ==========================================
# 3. TRAINING LOOP WITH GRADIENT PENALTY
# ==========================================
optimizer = optim.Adam(model.parameters(), lr=0.005)
mse_loss_fn = nn.MSELoss()

epochs = 1000
lambda_monotonic = 50.0  # Multiplier for monotonic violation penalty

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Enable autograd on inputs to calculate feature gradients (∂Price / ∂Features)
    X_input = X_tensor.clone().detach().requires_grad_(True)
    
    # Forward Pass
    predictions = model(X_input)
    
    # Standard MSE Prediction Loss
    base_loss = mse_loss_fn(predictions, y_tensor)

    # Compute Gradients of Predictions with respect to Inputs
    gradients = torch.autograd.grad(
        outputs=predictions,
        inputs=X_input,
        grad_outputs=torch.ones_like(predictions),
        create_graph=True,
        retain_graph=True
    )[0]

    # Monotonic Constraint: Penalize negative partial derivatives
    monotonic_violations = torch.relu(-gradients)
    monotonic_penalty = torch.mean(monotonic_violations ** 2)

    # Total Combined Loss
    total_loss = base_loss + (lambda_monotonic * monotonic_penalty)

    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Base Loss: {base_loss.item():.4f} | Monotonic Penalty: {monotonic_penalty.item():.6f}")

# ==========================================
# 4. EXPORT TO SINGLE-FILE ONNX FORMAT
# ==========================================
model.eval()
dummy_input = torch.randn(1, 5, dtype=torch.float32)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    dynamo=False  # Legacy TorchScript exporter forces all weights into a single .onnx file
)

print("Successfully exported single-file model to model.onnx!")
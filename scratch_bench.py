import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.resnet import Resnet
from src.models.tangos import Tangos
from src.models.batchensemble import BatchEnsemble
from src.models.ensemble import Ensemble

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

batch_size = 64
input_dim = 30
output_dim = 2
hidden_dim = 128
n_blocks = 3
dr = 0.2

crit = nn.CrossEntropyLoss()
x = torch.randn(batch_size, input_dim, device=device)
y = torch.randint(0, 2, (batch_size,), device=device)

# 1. Resnet
model1 = Resnet(True, input_dim, output_dim, hidden_dim, n_blocks, dr).to(device)
opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
for _ in range(5):
    opt1.zero_grad()
    loss = crit(model1(x), y)
    loss.backward()
    opt1.step()
if device == 'cuda': torch.cuda.synchronize()
t0 = time.perf_counter()
n_iters = 50
for _ in range(n_iters):
    opt1.zero_grad()
    loss = crit(model1(x), y)
    loss.backward()
    opt1.step()
if device == 'cuda': torch.cuda.synchronize()
t_resnet = (time.perf_counter() - t0) / n_iters * 1000
print(f"Resnet: {t_resnet:.2f} ms/batch", flush=True)

# 2. BatchEnsemble
backbone2 = Resnet(True, input_dim, output_dim, hidden_dim, n_blocks, dr)
model2 = BatchEnsemble(True, backbone2, ensemble_size=4, adapter_initialization="random_sign").to(device)
opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
for _ in range(5):
    opt2.zero_grad()
    loss = model2.training_loss(x, y, crit)
    loss.backward()
    opt2.step()
if device == 'cuda': torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(n_iters):
    opt2.zero_grad()
    loss = model2.training_loss(x, y, crit)
    loss.backward()
    opt2.step()
if device == 'cuda': torch.cuda.synchronize()
t_be = (time.perf_counter() - t0) / n_iters * 1000
print(f"BatchEnsemble: {t_be:.2f} ms/batch", flush=True)

# 3. Ensemble (lower triangular, cosine)
backbone3 = Resnet(True, input_dim, output_dim, hidden_dim, n_blocks, dr)
model3 = Ensemble(True, backbone3, ensemble_size=4, mask="lower_triangular", orthogonality_weight=0.1, additive_coef=0.5, similarity_metric="cosine").to(device)
opt3 = torch.optim.AdamW(model3.parameters(), lr=1e-3)
for _ in range(5):
    opt3.zero_grad()
    loss = crit(model3(x), y)
    loss.backward()
    opt3.step()
if device == 'cuda': torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(n_iters):
    opt3.zero_grad()
    loss = crit(model3(x), y)
    loss.backward()
    opt3.step()
if device == 'cuda': torch.cuda.synchronize()
t_ens = (time.perf_counter() - t0) / n_iters * 1000
print(f"Ensemble: {t_ens:.2f} ms/batch", flush=True)

# 4. Tangos
backbone4 = Resnet(True, input_dim, output_dim, hidden_dim, n_blocks, dr)
model4 = Tangos(True, backbone4, sparsity=1e-3, orthogonality=1e-2).to(device)
opt4 = torch.optim.AdamW(model4.parameters(), lr=1e-3)
for _ in range(2):
    opt4.zero_grad()
    loss = model4.training_loss(x, y, crit)
    loss.backward()
    opt4.step()
if device == 'cuda': torch.cuda.synchronize()
t0 = time.perf_counter()
n_iters_tangos = 10
for _ in range(n_iters_tangos):
    opt4.zero_grad()
    loss = model4.training_loss(x, y, crit)
    loss.backward()
    opt4.step()
if device == 'cuda': torch.cuda.synchronize()
t_tangos = (time.perf_counter() - t0) / n_iters_tangos * 1000
print(f"Tangos: {t_tangos:.2f} ms/batch", flush=True)

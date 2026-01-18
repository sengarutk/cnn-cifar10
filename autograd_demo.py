import torch

# Simple 2-layer NN
x = torch.randn(5, 10, requires_grad=True)
w1 = torch.randn(10, 20, requires_grad=True)
w2 = torch.randn(20, 1, requires_grad=True)

h = x @ w1
h_relu = torch.relu(h)
y = h_relu @ w2

loss = y.mean()
loss.backward()

print("Grad w1:", w1.grad.shape)
print("Grad w2:", w2.grad.shape)

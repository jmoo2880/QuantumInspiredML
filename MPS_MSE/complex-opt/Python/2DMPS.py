import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


def feature_map(x):
    # convert x to a pytorch tensor
    if not isinstance(x, torch.Tensor):
        x_tensor = torch.tensor(x, dtype=torch.cfloat)
    else:
        x_tensor = x.clone().detach()


    # calculate the two components of the feature map, s1 and s2
    s1 = torch.exp(1j * (3*torch.pi/2) * x_tensor) * torch.cos(torch.pi/2 * x_tensor)
    s2 = torch.exp(-1j * (3*torch.pi/2) * x_tensor) * torch.sin(torch.pi/2 * x_tensor)

    feature_vector = torch.stack((s1, s2), dim=-1)

    return feature_vector

class MPS(nn.Module):
    def __init__(self, d, chi):
        super(MPS, self).__init__()
        self.site1 = nn.Parameter(torch.randn(1, d, chi, dtype=torch.cfloat, requires_grad=True))
        self.site2 = nn.Parameter(torch.randn(chi, d, 1, dtype=torch.cfloat, requires_grad=True))
    
    def forward(self, p1, p2):
        # contract the sites with the mps
        #print(p1.shape)
        site_1_prodstate_1 = torch.einsum('ijk, j -> ik', self.site1, p1) # contract mps site 1 with product state site 1
        site_2_prodstate_2 = torch.einsum('ijk, j -> ik', self.site2, p2) # contract mps site 2 with product state site 2
        result = torch.einsum('ij, jk -> ik', site_1_prodstate_1, site_2_prodstate_2) # contract over the shared bond dimension chi
        # result is complex valued, take modulus
        result = result.abs().squeeze()

        return result


# set seed
torch.manual_seed(0)

# num_samples = 100
# class_A = torch.zeros((num_samples, 2), dtype=torch.float) # class A: 00
# class_B = torch.ones((num_samples, 2), dtype=torch.float) # class B: 11

# class_A_mapped = feature_map(class_A)
# class_A_mapped = feature_map(class_B)

# print(class_A_mapped.shape)

p1 = feature_map(1)
p2 = feature_map(1)

target = torch.tensor(1.0, dtype=torch.float)

# apply feature map
# class_A_mapped = feature_map(class_A)
# class_B_mapped = feature_map(class_B)

# # make labels
# targets_A = torch.zeros(num_samples, dtype=torch.float)
# targets_B = torch.ones(num_samples, dtype=torch.float) # aim to maximise overlap with 11

# data = torch.cat([class_A_mapped, class_B_mapped], dim=0)
# targets = torch.cat([targets_A, targets_B], dim=0)

# train the model
model = MPS(d=2, chi=4)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 300
for epoch in range(epochs):
    optimizer.zero_grad() # zero the gradient for the next loop 
    output = model(p1, p2)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# evaluate
print(model(p1, p2))

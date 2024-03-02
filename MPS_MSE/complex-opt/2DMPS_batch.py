import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import h5py

def feature_map(x):
    # convert x to a pytorch tensor
    if not isinstance(x, torch.Tensor):
        x_tensor = torch.tensor(x, dtype=torch.cfloat)
    else:
        x_tensor = x.clone().detach()


    # calculate the two components of the feature map, s1 and s2
    s1 = torch.exp(1j * (3*torch.pi/2) * x_tensor) * torch.cos(torch.pi/2 * x_tensor)
    s2 = torch.exp(-1j * (3*torch.pi/2) * x_tensor) * torch.sin(torch.pi/2 * x_tensor)

    feature_vector = torch.stack([s1, s2], dim=-1)

    return feature_vector

class MPS(nn.Module):
    def __init__(self, d, chi):
        super(MPS, self).__init__()
        self.site1 = nn.Parameter(torch.randn(1, d, chi, dtype=torch.cfloat, requires_grad=True))
        self.site2 = nn.Parameter(torch.randn(chi, d, 1, dtype=torch.cfloat, requires_grad=True))
    
    def forward(self, product_state):
        # contract the sites with the mps
        #print(p1.shape)
        # extract components
        p1 = product_state[:, 0, :] # batch dimension, site, vector component
        p2 = product_state[:, 1, :]
        site_1_prodstate_1 = torch.einsum('ijk, bj -> bik', self.site1, p1) # contract mps site 1 with product state site 1
        site_2_prodstate_2 = torch.einsum('ijk, bj -> bik', self.site2, p2) # contract mps site 2 with product state site 2
        result = torch.einsum('bij, bjk -> bik', site_1_prodstate_1, site_2_prodstate_2) # contract over the shared bond dimension chi
        # result is complex valued, take modulus
        result = result.abs().squeeze()

        return result


# set seed
torch.manual_seed(0)

print(feature_map(0.1))

num_samples = 100
class_A = torch.zeros((num_samples, 2), dtype=torch.float) # class A: 00
class_B = torch.ones((num_samples, 2), dtype=torch.float) # class B: 11

class_A_mapped = feature_map(class_A)
class_B_mapped = feature_map(class_B)

targets_A = torch.zeros(num_samples, dtype=torch.float)
targets_B = torch.ones(num_samples, dtype=torch.float) # aim to maximise overlap with 11

data = torch.cat([class_A_mapped, class_B_mapped], dim=0)
targets = torch.cat([targets_A, targets_B], dim=0)

model = MPS(d=2, chi=4)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad() # zero the gradient for the next loop 
    output = model(data)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


print(model.site1)
#rint(class_A_mapped[1, 0, :])

# get prediction accuracy

# class_A = torch.zeros((num_samples, 2), dtype=torch.float) # class A: 00
# class_B = torch.ones((num_samples, 2), dtype=torch.float) # class B: 11

# class_A_mapped = feature_map(class_A)
# class_B_mapped = feature_map(class_B)

# targets_A = torch.zeros(num_samples, dtype=torch.float)
# targets_B = torch.ones(num_samples, dtype=torch.float) # aim to maximise overlap with 11

# data_test = torch.cat([class_A_mapped, class_B_mapped], dim=0)
# targets_test = torch.cat([targets_A, targets_B], dim=0)

# preds_test = model(data_test)

# acc = torch.sum(preds_test == targets_test) / (num_samples * 2)
# print(f"Test accuracy: {acc}")
#print(model.site1)
#print(model.site2)

# try and save the tensors
# data_site1 = model.site1.detach().numpy()  # Your actual complex data for site1
# data_site2 = model.site2.detach().numpy()  # Your actual complex data for site2


# def complex_to_structured(arr):
#     structured_arr = np.empty(arr.shape, dtype=[('real', float), ('imag', float)])
#     structured_arr['real'] = arr.real
#     structured_arr['imag'] = arr.imag
#     return structured_arr

# structured_site1 = complex_to_structured(data_site1)
# structured_site2 = complex_to_structured(data_site2)

# with h5py.File('complex_data.h5', 'w') as hf:
#     hf.create_dataset('site1', data=structured_site1)
#     hf.create_dataset('site2', data=structured_site2)

#print(model.site1.detach().numpy().real)
#print(model.site2.detach().numpy())





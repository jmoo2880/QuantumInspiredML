using Zygote
using ITensors
using Random
using Plots

# make 2 site MPS site indices
num_sites = 2
s = siteinds("S=1/2", num_sites)

# set random seed for reproducibility
Random.seed!(42)

# make the mps, set χ = 2
mps = randomMPS(ComplexF64, s; linkdims=2)

# make a product state
ps = randomMPS(ComplexF64, s; linkdims=1)

# define our loss function as the negative of the modulus of the overlap
# this is a minimisation problem, so if we want to maximise the overlap, then we need to minimise the negative overlap
function loss(mps)
    res = 1
    for i=1:length(mps)
        res *= mps[i] * conj(ps[i])
    end
    return -abs(res[])
end

# main loop
lr = 0.1 # set the learning rate
num_steps = 100
println("Initial loss: $(loss(mps))")
loss_per_step = Vector{Float64}(undef, num_steps)
for step=1:num_steps
    # compute the gradient
    out, = gradient(loss, mps)
    update_site_1 = out[1][1]
    update_site_2 = out[1][2]
    # get the old sites 
    old_site_1 = deepcopy(mps[1])
    old_site_2 = deepcopy(mps[2])
    # apply the updates
    new_site_1 = old_site_1 - lr * update_site_1
    new_site_2 = old_site_2 - lr * update_site_2
    # add updated sites back into the mps
    mps[1] = new_site_1
    mps[2] = new_site_2
    # normalise the mps 
    normalize!(mps)
    # re-evaluate the loss
    new_loss = loss(mps)
    loss_per_step[step] = new_loss
    println("Step $step, loss = $new_loss")
end
plot(loss_per_step, xlabel="Number of steps", 
    ylabel="Negative Overlap", label="", title="2 Site MPS, lr=0.1")

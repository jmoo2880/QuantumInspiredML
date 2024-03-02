using Zygote
using ITensors
using Random
using Plots

Random.seed!(42)

struct PState
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate::MPS
    label::Int
end

# make 2 site MPS site indices
num_sites = 2
sites = siteinds("S=1/2", num_sites)
mps = randomMPS(ComplexF64, sites; linkdims=2)

# now let's define our product state encoding
function complex_feature_map(x::Float64)
    s1 = exp(1im * (3π/2) * x) * cospi(0.5 * x)
    s2 = exp(-1im * (2π/2) * x) * sinpi(0.5 * x)
    return [s1, s2]
end

# now generate class A and class B data - class A will be 00 and class B will be 11
class_A_samples = zeros(200, 2)
class_B_samples = ones(200, 2)
all_samples = vcat(class_A_samples, class_B_samples)
all_labels = Int.(vcat(zeros(size(class_A_samples)[1]), ones(size(class_B_samples)[1])))

# map to product states
function DataToProductState(sample::Vector, site_inds::Vector{Index{Int64}})
    n_sites = length(site_inds)
    product_state = MPS(ComplexF64, site_inds; linkdims=1)
    for j=1:n_sites
        T = ITensor(site_inds[j])
        zero_state, one_state = complex_feature_map(sample[j])
        T[1] = zero_state
        T[2] = one_state
        product_state[j] = T 
    end
    return product_state
end

all_product_states = Vector{PState}(undef, 400)
for p = 1:400
    sample_pstate = DataToProductState(all_samples[p, :], sites)
    sample_label = all_labels[p]
    product_state = PState(sample_pstate, sample_label)
    all_product_states[p] = product_state
end

function LossPerSample(mps, ps)
    y = ps.label # 1 or 0
    yhat = 1
    for i=1:length(mps)
        yhat *= mps[i] * conj(ps.pstate[i])
    end
    # take the modulus
    final_yhat = abs(yhat[])
    diff_sq = (final_yhat - y)^2
    loss = 0.5 * diff_sq
    return loss
end

function LossAndGradient(mps, all_pstates)
    site1_gradient = []
    site2_gradient = []
    for i=1:length(all_pstates)
        grad = gradient(LossPerSample, mps, all_pstates[i])
        push!(site1_gradient, grad[1][1][1])
        push!(site2_gradient, grad[1][1][2])
    end
    return site1_gradient, site2_gradient
end

function LossPerDataset(mps, all_pstates)
    loss_total = 0
    for i=1:length(all_pstates)
        loss = LossPerSample(mps, all_pstates[i])
        loss_total += loss
    end
    loss_final = loss_total/length(all_pstates)
    return loss_final
end

lr = 0.4
num_steps = 500
init_loss = LossPerDataset(mps, all_product_states)
loss_store = Vector{Float64}(undef, num_steps)
for step=1:num_steps
    s1_g, s2_g = LossAndGradient(mps, all_product_states)
    overall_s1_grad = sum(s1_g)./200
    overall_s2_grad = sum(s2_g)./200
    s1_old = deepcopy(mps[1])
    s2_old = deepcopy(mps[2])

    s1_new = s1_old - lr * overall_s1_grad
    s2_new = s2_old - lr * overall_s2_grad
    mps[1] = s1_new
    mps[2] = s2_new
    normalize!(mps)
    new_loss = LossPerDataset(mps, all_product_states)
    loss_store[step] = new_loss
    println("Step $step, loss = $new_loss")
end


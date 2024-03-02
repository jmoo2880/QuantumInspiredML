using Zygote
using ITensors
using Random
using Plots 
using Base.Threads

Random.seed!(42)

struct PState
    pstate::MPS
    label::Int
end

num_sites = 2
sites = siteinds("S=1/2", num_sites)
mps = randomMPS(ComplexF64, sites; linkdims=2)

function complex_feature_map(x::Float64)
    s1 = exp(1im * (3π/2) * x) * cospi(0.5 * x)
    s2 = exp(-1im * (2π/2) * x) * sinpi(0.5 * x)
    return [s1, s2]
end

function generate_training_data(samples_per_class::Int)

    class_A_samples = zeros(samples_per_class, 2)
    class_B_samples = ones(samples_per_class, 2)
    all_samples = vcat(class_A_samples, class_B_samples)
    all_labels = Int.(vcat(zeros(size(class_A_samples)[1]), ones(size(class_B_samples)[1])))

    return all_samples, all_labels

end

function make_prediction(mps, ps)
    # make a prediction a single product state
    res = 1
    for i in eachindex(mps)
        res *= mps[i] * ps.pstate[i]
    end
    yhat = abs(res[]) # modulus of overlap
    # binary classifier so use a threshold of 0.5 to make predictions
    # since we trained to maximise the overlap with class B (11) which was assigned the 1 label, we return 1 if overlap > 0.5. 
    pred = 0
    if yhat > 0.5
        pred = 1
    end
    return pred
end

function accuracy_per_dataset(mps, all_pstates)
    running_correct = 0
    for i in eachindex(all_product_states)
        model_pred = make_prediction(mps, all_pstates[i])
        if model_pred == all_pstates[i].label
            running_correct += 1
        end
    end
    return running_correct/length(all_pstates)
end

# map to product states
function sample_to_product_state(sample::Vector, site_inds::Vector{Index{Int64}})
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

function dataset_to_product_state(dataset::Matrix, labels::Vector, sites::Vector{Index{Int64}})

    all_product_states = Vector{PState}(undef, size(dataset)[1])
    for p=1:length(all_product_states)
        sample_pstate = DataToProductState(dataset[p, :], sites)
        sample_label = labels[p]
        product_state = PState(sample_pstate, sample_label)
        all_product_states[p] = product_state
    end

    return all_product_states

end

function loss_per_sample(mps, ps)
    y = ps.label
    yhat = 1
    for i in eachindex(mps)
        yhat *= mps[i] * conj(ps.pstate[i])
    end
    # take the modulus
    final_yhat = abs(yhat[])
    diff_sq = (final_yhat - y)^2
    loss = 0.5 * diff_sq
    return loss
end

function loss_per_dataset(mps, all_pstates)
    total_loss = 0
    for i in eachindex(all_pstates)
        total_loss += loss_per_sample(mps, all_pstates[i])
    end

    return total_loss / length(all_pstates)
end

function loss_and_gradients_per_sample(mps, ps)
    loss, (∇,) = withgradient(loss_per_sample, mps, ps)
    ∇1 = ∇[:data][1]
    ∇2 = ∇[:data][2]
    return loss, ∇1, ∇2
end

function loss_and_gradient_dataset(mps, all_pstates)
    ∇1_accum = Vector{ITensor}(undef, length(all_product_states))
    ∇2_accum = Vector{ITensor}(undef, length(all_product_states))
    loss_vals = Vector{Float64}(undef, length(all_product_states))
    Threads.@threads for i in eachindex(all_pstates)
        loss, ∇1, ∇2 = loss_and_gradients_per_sample(mps, all_pstates[i])
        loss_vals[i] = loss
        ∇1_accum[i] = ∇1
        ∇2_accum[i] = ∇2
    end

    loss_final = sum(loss_vals) / length(all_pstates)
    ∇1_final = sum(∇1_accum) ./ length(all_pstates)
    ∇2_final = sum(∇2_accum) ./ length(all_pstates)

    return loss_final, ∇1_final, ∇2_final

end


all_samples, all_labels = generate_training_data(200)
all_product_states = dataset_to_product_state(all_samples, all_labels, sites)
shuffle!(all_product_states)

num_steps = 500
lr = 0.5
init_loss = loss_per_dataset(mps, all_product_states)
init_acc = accuracy_per_dataset(mps, all_product_states)
println("Initial loss: $init_loss | initial acc: $init_acc")
running_loss = []
push!(running_loss, init_loss)
for step=1:num_steps
    # get the gradient
    loss, ∇1, ∇2 = loss_and_gradient_dataset(mps, all_product_states)
    old_site_1 = deepcopy(mps[1])
    old_site_2 = deepcopy(mps[2])

    new_site_1 = old_site_1 - lr * ∇1
    new_site_2 = old_site_2 - lr * ∇2
    
    # place new sites into the mps
    mps[1] = new_site_1
    mps[2] = new_site_2

    normalize!(mps)


    # re-evaluate loss 
    new_loss = loss_per_dataset(mps, all_product_states)
    push!(running_loss, new_loss)
    println("Step $step, loss = $new_loss")
end


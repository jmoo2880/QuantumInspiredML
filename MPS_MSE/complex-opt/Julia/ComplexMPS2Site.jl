using Zygote
using ITensors
using Random
using Plots 
using Base.Threads
using Distributions

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

function generate_mv_training_data(samples_per_class::Int)
    mean_A = [0.3, 0.7] # upper left of unit square
    cov_A = [0.02 0.01; 0.01 0.02]
    dist_A = MvNormal(mean_A, cov_A)

    mean_B = [0.7, 0.3]
    cov_B = [0.01 0.005; 0.005 0.01]
    dist_B = MvNormal(mean_B, cov_B)

    samples_A = Matrix{Float64}(undef, 0, 2)  # Initialize as an empty 0x2 matrix
    samples_B = Matrix{Float64}(undef, 0, 2)  # Initialize as an empty 0x2 matrix
    
    while size(samples_A, 1) < samples_per_class
        temp_samples_A = transpose(rand(dist_A, samples_per_class - size(samples_A, 1)))
        filtered_samples_A = filter(x -> 0 <= x[1] <= 1 && 0 <= x[2] <= 1, eachrow(temp_samples_A))
        for sample in filtered_samples_A
            samples_A = vcat(samples_A, reshape(sample, 1, length(sample)))  # Reshape each sample to a 1x2 row
        end
    end
    
    while size(samples_B, 1) < samples_per_class
        temp_samples_B = transpose(rand(dist_B, samples_per_class - size(samples_B, 1)))
        filtered_samples_B = filter(x -> 0 <= x[1] <= 1 && 0 <= x[2] <= 1, eachrow(temp_samples_B))
        for sample in filtered_samples_B
            samples_B = vcat(samples_B, reshape(sample, 1, length(sample)))  # Reshape each sample to a 1x2 row
        end
    end

    all_samples = vcat(samples_A, samples_B)
    all_labels = Int.(vcat(zeros(size(samples_A)[1]), ones(size(samples_B)[1]))) # maximise overlap with class B
    
    return all_samples, all_labels

end


function make_prediction(mps, ps)
    # make a prediction a single product state
    res = 1
    for i in eachindex(mps)
        res *= mps[i] * conj(ps.pstate[i])
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
        sample_pstate = sample_to_product_state(dataset[p, :], sites)
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


all_samples, all_labels = generate_mv_training_data(500)
all_product_states = dataset_to_product_state(all_samples, all_labels, sites)
shuffle!(all_product_states)

num_steps = 500
lr = 0.8
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
println("Final test accuracy: $(accuracy_per_dataset(mps, all_product_states))")
all_test_samples, all_test_labels = generate_mv_training_data(500)
all_test_product_states = dataset_to_product_state(all_test_samples, all_test_labels, sites)
println("Test accuracy: $(accuracy_per_dataset(mps, all_test_product_states))")

# not efficient, I know... 
function SampleMPS(num_samples)
    # do site 1 first
    samples = Matrix{Float64}(undef, num_samples, 2)
    xs = collect(0:0.01:1)
    states = [complex_feature_map(x) for x in xs]
    mps_copy_site_1 = deepcopy(mps)
    orthogonalize!(mps_copy_site_1, 1)

    ρ1 = prime(mps_copy_site_1[1], sites[1]) * dag(mps_copy_site_1[1])
    ρ1 = matrix(ρ1)

    # check traces are equal to 1
    if !isequal(abs(tr(ρ1)), 1)
        error("Trace of 1-site RDM at site 1 not equal to 1")
    end
    # get proba densities
    proba_densities_site1 = []
    for j in eachindex(states)
        psi = states[j]
        expect_s1 = abs(psi' * ρ1 * psi)^2
        push!(proba_densities_site1, expect_s1)
    end
    # normalise 
    proba_densities_site1 ./= sum(proba_densities_site1)
    # check 
    if !isequal(sum(proba_densities_site1), 1)
        error("Site 1 distribution not normalized!")
    end
    cdf_site1 = cumsum(proba_densities_site1)
    # now sample
    for i in eachindex(samples)
        # start with site 1
        r = rand()
        k1 = findlast(x -> x <= r, cdf_site1)
        #println("Sampled state: $(states[k1]) -> x = $(xs[k1])")
        samples[i, 1] = xs[k1]
        # now construct projector and project mps into subspace
        sampled_state = xs[k1]
        site_1_projector = sampled_state * sampled_state'
        # make into a one site MPO 
        site_1_proj_op = op(site_1_projector, sites[1])
        # apply to the mps at site 1
        site_1_old = deepcopy(mps[1])
        site_1_new = site_1_old * site_1_proj_op
        noprime!(site_1_new)
        # add back into the mps and normalise
        mps[1] = site_1_new
        normalize!(mps)
        # now sample site 2, conditioned on first sample
        mps_copy_site_2 = deepcopy(mps)
        orthogonalize!(mps_copy_site_2, 2)
        ρ2 = prime(mps_copy_site_2[2], sites[2]) * dag(mps_copy_site_1[2])
        ρ2 = matrix(ρ2)
        # do checks
        if !isequal(abs(tr(ρ2)), 1)
            error("Trace of 1 site RDM at site 2 not equal to 1.")
        end

        proba_densities_site2 = []
        for j in eachindex(states)
            psi = states[j]
            expect_s2 = abs(psi' * ρ2 * psi)^2
            push!(proba_densities_site2, expect_s2)
        end
        # normalise 
        proba_densities_site2 ./= sum(proba_densities_site2)
        # check 
        if !isequal(sum(proba_densities_site2), 1)
            error("Site 1 distribution not normalized!")
        end
        cdf_site2 = cumsum(proba_densities_site2)
        r = rand()
        k2 = findlast(x -> x <= r, cdf_site2)
        #println("Sampled state: $(states[k2]) -> x = $(xs[k2])")
        samples[i, 2] = xs[k2]

        println("Sample $(i): [$(xs[k1]), $(xs[k2])]")
    end

end
# sampling
# mps_copy = deepcopy(mps)
# orthogonalize!(mps_copy, 1)
# ρ1 = prime(mps_copy[1], sites[1]) * dag(mps_copy[1])
# ρ1 = matrix(ρ1)
# # consruct prob interval
# xs = collect(0:0.01:1)
# states = [complex_feature_map(x) for x in xs]
# proba_densities = []
# for i in eachindex(states)
#     psi = states[i]
#     expect = abs(psi' * ρ1 * psi)^2
#     push!(proba_densities, expect)
# end
# proba_densities_norm = proba_densities ./ sum(proba_densities)
# r = rand() # generate uniform random value
# cdf = cumsum(proba_densities_norm)
# k = findlast(x -> x <= r, cdf)
# println("Sampled state: $(states[k]) -> x = $(xs[k])")
# # construct projector
# sampled_state = states[k]
# projector_site_1 = sampled_state * sampled_state'
# projector_site_1_op = op(projector_site_1, sites[1])
# # apply to mps 
# site_1_old = deepcopy(mps[1])
# site_1_new = site_1_old * projector_site_1_op
# noprime!(site_1_new)
# mps[1] = site_1_new
# normalize!(mps)

# # site 2
# mps_copy2 = deepcopy(mps)
# orthogonalize!(mps_copy2, 2)
# ρ2 = prime(mps_copy2[2], sites[2]) * dag(mps_copy2[2])
# ρ2 = matrix(ρ2)
# proba_densities2 = []
# for i in eachindex(states)
#     psi = states[i]
#     expect = abs(psi' * ρ2 * psi)^2
#     push!(proba_densities2, expect)
# end
# proba_densities_norm2 = proba_densities2 ./ sum(proba_densities2)
# r2 = rand() # generate uniform random value
# cdf2 = cumsum(proba_densities_norm2)
# k2 = findlast(x -> x <= r2, cdf2)
# println("Sampled state: $(states[k2]) -> x = $(xs[k2])")

# println("Sampled state (x1, x2): ($(xs[k]), $(xs[k2]))")


using Plots
using Plots.PlotMeasures
using StatsPlots
using Base.Threads
using QuadGK
using JLD2
include("../../LogLoss/RealRealHighDimension.jl");

x = legendre_encode_no_norm(0.1, 3)

proba_density(state::Vector, rdm::Matrix) = abs(state' * rdm * state)
function proba_density(x::Float64, rdm::Matrix, encoding::Symbol, d::Int)
    enc = model_encoding(encoding)
    state = enc.encode(x, d)
    return abs(state' * rdm * state)
end

function norm_constant(rdm::Matrix, range::Tuple, encoding::Symbol, d::Int)
    if first(range) > last(range)
        upper, lower = range
    else
        lower, upper = range
    end
    pdf(x) = proba_density(x, rdm, encoding, d)
    Z, _ = quadgk(pdf, lower, upper)
    return Z
end

function conditional_probability_mean(rdm::Matrix, encoding::Symbol, d::Int; dx=1E-4)
    enc = model_encoding(encoding)
    range = enc.range
    Z = norm_constant(rdm, range, encoding, d)
    lower, upper = range
    xvals = collect(lower:dx:upper)
    probs = Vector{Float64}(undef, length(xvals))
    for (index, val) in enumerate(xvals)
        prob = (1/Z) * proba_density(val, rdm, encoding, d)
        probs[index] = prob
    end

    expect_x = sum(xvals .* probs) * dx
    return expect_x
end

function encoding_mean_uncertainty(encoding::Symbol, d::Int64; num_eval_pts=1000)
    enc = model_encoding(encoding)
    r = enc.range
    xprs = LinRange(first(r), last(r), num_eval_pts)
    states = enc.encode.(xprs, d)
    means = zeros(length(xprs))
    for (i, st_gt) in enumerate(states)
        rdm = st_gt * st_gt'
        expect_x = conditional_probability_mean(rdm, encoding, d)
        means[i] = expect_x
    end
    errs = abs.(xprs - means)
    return errs
end

function encoding_mode_uncertainty(encoding::Symbol, d::Int64; num_eval_pts=1000)
    # make the encoding
    enc = model_encoding(encoding)
    # get ranges
    r = enc.range
    # specify xprimes
    xprs = LinRange(first(r), last(r), num_eval_pts)
    # convert to states
    states = enc.encode.(xprs, d)
    modes = zeros(length(xprs))
    for (i, st_gt) in enumerate(states)
        # fix the rdm
        rdm = st_gt * st_gt'
        # now that we have the ground truth rdm, we loop over probe states to get probas
        pdf(x) = proba_density(x, rdm)
        probas = pdf.(states)
        # get the mode
        mode_idx = argmax(probas)
        mode_x = xprs[mode_idx]
        modes[i] = mode_x
    end

    errs = abs.(xprs - modes)

    return errs

end

function inspect_state(x::Float64, encoding::Symbol, d::Int64; 
    num_evals::Int=1000)
    # inspect the probability density function for a given RDM
    # and encoding with the mode overlayed for reference
    enc = model_encoding(encoding)
    r = enc.range
    gt_state = enc.encode(x, d)
    gt_rdm = gt_state * gt_state'
    xprimes = LinRange(first(r), last(r), num_evals)
    xprime_states = enc.encode.(xprimes, d)
    # map primes
    probadens(x) = proba_density(x, gt_rdm)
    xprime_probas = probadens.(xprime_states)
    # get the mode
    mode_idx = argmax(xprime_probas)
    mode_val = xprimes[mode_idx]
    abs_err = abs(mode_val - x)
    
    p = plot(xprimes, xprime_probas,
        xlabel="x'",
        ylabel="p(x')", 
        label="",
        title="$(enc.name), x=$x, d=$d\nAbs(Err) = $(round(abs_err; digits=4))",
        lw = 2)
    vline!([mode_val], ls=:dot, lw=2, label="Mode")
    vline!([x], ls=:dash, lw=2, label="x")
    display(p)

    return xprime_probas

end

function plot_mean_mode_uncertainty(encoding::Symbol, d::Int; num_eval_pts::Int=1000)
    # plot both the mean and mode uncertainty together
    # plot the difference to identify in which regions the mean vs mode would be optimal
    enc = model_encoding(encoding)
    r = enc.range
    mean_errs = encoding_mean_uncertainty(encoding, d; num_eval_pts=num_eval_pts)
    mode_errs = encoding_mode_uncertainty(encoding, d; num_eval_pts=num_eval_pts)
    xprs = LinRange(first(r), last(r), num_eval_pts)
    p1 = plot(xprs, mean_errs, label="Mean", legend=:outertopright, 
        xlabel="x'", ylabel="Abs(Error)", title="$(enc.name), d=$d")
    plot!(xprs, mode_errs, label="Mode")
    
    diff_errs = mode_errs .- mean_errs
    p2 = plot(xprs, diff_errs, xlabel="x'", ylabel="Mode - Mean", label="")
    hline!([0.0], ls=:dot, c=:black, label="")

    # expected error if using mean/mode
    best_err = min.(mode_errs, mean_errs)
    p3 = plot(xprs, best_err, xlabel="x'", ylabel="Abs(Error)", label="Mean/Mode Switching",
        ylims=(0, maximum(vcat(mean_errs, mode_errs))))
    #plot!(xprs, mode_errs, label="Mode Only")
    
    p = plot(p1, p2, p3, size=(1200, 800), bottom_margin=5mm, left_margin=5mm)


    display(p)
    
end

function plot_mode_uncert_heatmap(encoding::Symbol)
    ds = 2:12
    num_eval_pts = 1000
    enc = model_encoding(encoding)
    r = enc.range
    xs = LinRange(first(r), last(r), num_eval_pts)
    d_errors = Matrix{Float64}(undef, length(ds), num_eval_pts)
    @threads for i in eachindex(ds)
        errs = encoding_mode_uncertainty(encoding, ds[i]; num_eval_pts=num_eval_pts)
        d_errors[i, :] = errs
    end 
    p1 = heatmap(xs, ds, d_errors, 
        xlabel="x'", ylabel="d", 
        title="$(enc.name) Mode Uncertainty", 
        yticks=(2:12), xticks=(first(r):0.2:last(r)),
        c=:viridis, clims=(0.0, 1.0), colorbar_title="Abs(Error)",
        )

    p2 = plot(xlabel="x'", ylabel="Abs(Error)", legend=:outertopright, title="d slices")
    for d in ds
        plot!(xs, d_errors[(d-1), :], label="d=$(d)")
    end
    p8 = plot(xs, d_errors[7, :], xlabel="x'", ylabel="Abs(Error)", label="d = 8", legend=:outertopright)
    p9 = plot!(xs, d_errors[8, :], label="d = 9")
    p10 = plot!(xs, d_errors[9, :], label="d = 10")
    p11 = plot!(xs, d_errors[10, :], label="d = 11")
    p12 = plot!(xs, d_errors[11, :], label="d = 12")

    p = plot(p1, p2, p8, size=(1000, 500), bottom_margin=5mm, left_margin=5mm)
    
    display(p)

end


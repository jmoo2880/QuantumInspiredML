using Plots
using Plots.PlotMeasures
using StatsPlots
using Base.Threads
using QuadGK
using JLD2
include("../../LogLoss/RealRealHighDimension.jl");

x = legendre_encode_no_norm(0.1, 3)

proba_density(state::Vector, rdm::Matrix) = abs(state' * rdm * state)

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


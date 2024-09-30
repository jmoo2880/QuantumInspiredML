using Plots
using Plots.PlotMeasures
using StatsPlots
using Roots
using Base.Threads
using QuadGK
using JLD2
include("../../LogLoss/RealRealHighDimension.jl");

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

function cdf(x::Float64, rdm::Matrix, Z::Float64, encoding::Symbol, d::Int)
    prob_density_wrapper(x_prime) = (1/Z) * proba_density(x_prime, rdm, encoding, d)
    lower, _ = model_encoding(encoding).range
    cdf_val, _ = quadgk(prob_density_wrapper, lower, x)
    return cdf_val
end

function conditional_probability_median(rdm::Matrix, encoding::Symbol, d::Int; dx=1E-4)
    enc = model_encoding(encoding)
    lower, upper = enc.range
    Z = norm_constant(rdm, (lower, upper), encoding, d)
    u = 0.5 # find the value of xprime for which the cdf is 0.5 (i.e. the median)
    cdf_wrapper(x) = cdf(x, rdm, Z, encoding, d) - u
    median_x = find_zero(cdf_wrapper, (lower, upper); rtol=0.0)
    return median_x
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

# function encoding_median_uncertainty(encoding::Symbol, d::Int64; num_eval_pts=1000)
#     enc = model_encoding(encoding)

# end

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

function encoding_median_uncertainty(encoding::Symbol, d::Int64; num_eval_pts=1000)
    enc = model_encoding(encoding)
    # get ranges
    r = enc.range
    # specify xprimes
    xprs = LinRange(first(r), last(r), num_eval_pts)
    # convert to states
    states = enc.encode.(xprs, d)
    medians = zeros(length(xprs))
    for (i, st_gt) in enumerate(states)
        # fix the rdm
        rdm = st_gt * st_gt'
        median_x = conditional_probability_median(rdm, encoding, d)
        medians[i] = median_x
    end
    errs = abs.(xprs - medians) 
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
    # get the mean/expectation
    rdm = gt_state * gt_state'
    expect_x = conditional_probability_mean(rdm, encoding, d)
    # get the median
    median_x = conditional_probability_median(rdm, encoding, d)

    p1 = plot(xprimes, xprime_probas,
        xlabel="x'",
        ylabel="p(x')", 
        label="",
        title="$(enc.name), x=$x, d=$d\nAbs(Err) = $(round(abs_err; digits=4))",
        lw = 2)
    vline!([x], lw=2, label="x")
    vline!([mode_val], ls=:dash, lw=2, label="Mode")
    vline!([median_x], ls=:dash, lw=2, label="Median")
    vline!([expect_x], ls=:dash, lw=2, label="Mean")

    p2 = plot(xprimes, xprime_probas,
        xlabel="x'",
        ylabel="p(x')", 
        label="",
        title="$(enc.name), x=$x, d=$d\nAbs(Err) = $(round(abs_err; digits=4))",
        lw = 2, xlims=((x-0.3), (x+0.3)))
    vline!([x], lw=3, label="x")
    vline!([mode_val], ls=:dash, lw=2, label="Mode")
    vline!([median_x], ls=:dash, lw=2, label="Median")
    vline!([expect_x], ls=:dash, lw=2, label="Mean")

    p = plot(p1, p2, bottom_margin=5mm, left_margin=5mm, top_margin=5mm, size=(1200, 400))
    display(p)

    return xprime_probas

end

function plot_uncertainty_per_d(encoding::Symbol, d::Int; num_eval_pts::Int=1000)
    # plot the mean, median and mode uncertainty together
    enc = model_encoding(encoding)
    r = enc.range
    mean_errs = encoding_mean_uncertainty(encoding, d; num_eval_pts=num_eval_pts)
    mode_errs = encoding_mode_uncertainty(encoding, d; num_eval_pts=num_eval_pts)
    median_errs = encoding_median_uncertainty(encoding, d; num_eval_pts=num_eval_pts)
    xprs = LinRange(first(r), last(r), num_eval_pts)
    #p1 = plot(xprs, mean_errs, label="Mean", legend=:outertopright, 
    #    xlabel="x'", ylabel="Abs(Error)", title="$(enc.name), d=$d")
    p1 = plot(xprs, mode_errs, label="Mode", xlabel="x'", ylabel="Abs(Error)", title="$(enc.name), d=$d")
    plot!(xprs, median_errs, label="Median")
    
    diff_errs_mode_mean = mode_errs .- mean_errs
    p2 = plot(xprs, diff_errs_mode_mean, xlabel="x'", ylabel="Mode - Mean", label="")
    hline!([0.0], ls=:dot, c=:black, label="")

    diff_errs_mode_median = mode_errs .- median_errs
    p3 = plot(xprs, diff_errs_mode_median, xlabel="x'", ylabel="Mode - Median", label="")
    hline!([0.0], ls=:dot, c=:black, label="")

    # expected error if using mean/mode
    best_err = min.(mode_errs, mean_errs)
    best_err_med_mod = min.(mode_errs, median_errs)
    p4 = plot(xprs, best_err, xlabel="x'", ylabel="Abs(Error)", label="Mean/Mode Switching",
        ylims=(0, maximum(vcat(mean_errs, mode_errs))))
    plot!(xprs, best_err_med_mod, label="Median/Mode Switching")
    #plot!(xprs, mode_errs, label="Mode Only")
    
    p = plot(p1, p2, p3, p4, size=(1200, 800), bottom_margin=5mm, left_margin=5mm)

    display(p)
    
end

function plot_mean_mode_median_uncert_heatmap(encoding::Symbol)
    ds = 6:20
    num_eval_pts = 500
    enc = model_encoding(encoding)
    r = enc.range
    xs = LinRange(first(r), last(r), num_eval_pts)
    mode_errs = Matrix{Float64}(undef, length(ds), num_eval_pts)
    mean_errs = Matrix{Float64}(undef, length(ds), num_eval_pts)
    median_errs = Matrix{Float64}(undef, length(ds), num_eval_pts)
    @threads for i in eachindex(ds)
        mode_errs[i, :] = encoding_mode_uncertainty(encoding, ds[i]; num_eval_pts=num_eval_pts)
        mean_errs[i, :] = encoding_mean_uncertainty(encoding, ds[i]; num_eval_pts=num_eval_pts)
        median_errs[i, :] = encoding_median_uncertainty(encoding, ds[i]; num_eval_pts=num_eval_pts)
    end
    # determine clims based on mean errors (larger)
    min_clim, max_clim = minimum(mean_errs), maximum(mean_errs)
    cmap = cgrad(:cork, scale=:exp)
    p1 = heatmap(xs, collect(ds), mode_errs, xlabel="x'", ylabel='d', title="Mode Error", clim=(min_clim, max_clim), cmap=cmap,
        colorbar_title="Abs(Error)")
    p2 = heatmap(xs, collect(ds), mean_errs, xlabel="x'", ylabel='d', title="Mean Error", clim=(min_clim, max_clim), cmap=cmap,
        colorbar_title="Abs(Error)")
    diff_errs = mode_errs - mean_errs
    p3 = heatmap(xs, collect(ds), median_errs, xlabel="x'", ylabel='d', title="Median Error", clim=(min_clim, max_clim), cmap=cmap,
        colorbar_title="Abs(Error)")
    p4 = heatmap(xs, collect(ds), diff_errs, xlabel="x'", ylabel='d', title="Diff (Mode - Mean)", cmap=:vik,
        clim=(-maximum(abs.(diff_errs)), maximum(abs.(diff_errs))))
    diff_mode_median = mode_errs - median_errs
    p5 = heatmap(xs, collect(ds), diff_mode_median, xlabel="x'", ylabel='d', title="Diff (Mode - Median)", cmap=:vik,
        clim=(-maximum(abs.(diff_errs)), maximum(abs.(diff_errs))))

    best_errs = min.(mode_errs, mean_errs)
    p6 = heatmap(xs, collect(ds), best_errs, xlabel="x'", ylabel='d', title="Mean/Mode Switching Error", clim=(min_clim, max_clim),
        cmap=cmap, colorbar_title="Abs(Error)")
    
    best_med_mod = min.(mode_errs, median_errs)
    p7 = heatmap(xs, collect(ds), best_med_mod, xlabel="x'", ylabel='d', title="Median/Mode Switching Error", clim=(min_clim, max_clim),
        cmap=cmap, colorbar_title="Abs(Error)")

    diff_two_switching = best_med_mod - best_errs
    p8 = heatmap(xs, collect(ds), diff_two_switching, xlabel="x'", ylabel='d', title="Diff (MED/MOD - MEA/MOD)", cmap=:vik, 
        clim=(-maximum(abs.(diff_two_switching)), maximum(abs.(diff_two_switching))))

    p = plot(p1, p2, p3, p4, p5, p6, p7, p8, size=(1500, 1000), bottom_margin=5mm, left_margin=5mm, layout=(3, 3))
    display(p)
end

function plot_avg_error_domain(encoding::Symbol)
    # for each value of d, compute avg. error across the encoding domain [-1, 1]
    # to see whether mode only, mean only, or mean/mode switching is better for a given d
    ds = 4:20
    pal = palette(:tab10)
    num_eval_pts = 200
    enc = model_encoding(encoding)
    r = enc.range
    xs = LinRange(first(r), last(r), num_eval_pts)
    mode_errs = Matrix{Float64}(undef, length(ds), num_eval_pts)
    mean_errs = Matrix{Float64}(undef, length(ds), num_eval_pts)
    median_errs = Matrix{Float64}(undef, length(ds), num_eval_pts)
    @threads for i in eachindex(ds)
        mode_errs[i, :] = encoding_mode_uncertainty(encoding, ds[i]; num_eval_pts=num_eval_pts)
        mean_errs[i, :] = encoding_mean_uncertainty(encoding, ds[i]; num_eval_pts=num_eval_pts)
        median_errs[i, :] = encoding_median_uncertainty(encoding, ds[i]; num_eval_pts=num_eval_pts)
    end
    mode_avg_err_per_d = mean(mode_errs, dims=2)
    mean_avg_err_per_d = mean(mean_errs, dims=2)
    median_avg_err_per_d = mean(median_errs, dims=2)

    # mean/mode switching
    best = min.(mode_errs, mean_errs)
    best_avg_err_per_d = mean(best, dims=2)

    # median/mode switching
    best_med_mo = min.(mode_errs, median_errs)
    best_avg_err_per_d_med_mo = mean(best_med_mo, dims=2)

    # mode - mode/mean-switching diff
    mode_mode_sw_diff_abs = (best_avg_err_per_d - mode_avg_err_per_d)
    mode_mode_sw_diff_rel = (abs.((best_avg_err_per_d) - (mode_avg_err_per_d))) ./  mode_avg_err_per_d

    # mean - median-switching
    mean_median_sw = min.(median_errs, mean_errs)
    avg_err_mean_median_sw = mean(mean_median_sw, dims=2)


    p1 = plot(ds, mean_avg_err_per_d, c=pal[1], label="", lw=1, title="Legendre Encoding", xticks=collect(ds), alpha=0.7)
    scatter!(ds, mean_avg_err_per_d, label="Mean Only", xlabel="d", ylabel="Mean Error ∈ [-1, 1]", c=pal[1], alpha=0.7)
    plot!(ds, mode_avg_err_per_d,label="", c=pal[2], lw=1, alpha=0.7)
    scatter!(ds, mode_avg_err_per_d, label="Mode Only", c=pal[2], alpha=0.7)
    scatter!(ds, median_avg_err_per_d, label="Median Only", c=pal[3], alpha=0.7)
    plot!(ds, median_avg_err_per_d, label="", c=pal[3], lw=1, alpha=0.7)
    scatter!(ds, best_avg_err_per_d, label="Mean/Mode Switching", c=pal[4], alpha=0.7)
    plot!(ds, best_avg_err_per_d, label="", c=pal[4], lw=1, alpha=0.7)
    scatter!(ds, best_avg_err_per_d_med_mo, label="Median/Mode Switching", c=pal[5], alpha=0.7)
    plot!(ds, best_avg_err_per_d_med_mo, label="", c=pal[5], lw=1, alpha=0.7)
    scatter!(ds, avg_err_mean_median_sw, label="Mean/Median Switching", c=pal[6], alpha=0.7)
    plot!(ds, avg_err_mean_median_sw,label="", c=pal[6], alpha=0.7, lw=1)

    p2 = plot(ds, mode_mode_sw_diff_abs, xlabel="d", ylabel="abs. error w.r.t. mode only", label="", c=:black, 
        title="Mean/Mode Switching Advantage", xticks=collect(ds))
    scatter!(ds, mode_mode_sw_diff_abs, c=:black, label="")
    p3 = plot(ds, mode_mode_sw_diff_rel, xlabel="d", ylabel="rel. error w.r.t. mode only", label="", c=:black, 
        title="Mean/Mode Switching Advantage", xticks=collect(ds))
    scatter!(ds, mode_mode_sw_diff_rel, c=:black, label="")

    #p = plot(p1, p2, p3, size=(1500, 800), bottom_margin=5mm, left_margin=5mm)
    p = plot(p1, size=(500, 300))
    display(p)
end

function plot_median_uncert_heatmap(encoding::Symbol)
    ds = 6:20
    num_eval_pts = 1000
    enc = model_encoding(encoding)
    r = enc.range
    xs = LinRange(first(r), last(r), num_eval_pts)
    d_errors = Matrix{Float64}(undef, length(ds), num_eval_pts)
    @threads for i in eachindex(ds)
        errs = encoding_median_uncertainty(encoding, ds[i]; num_eval_pts=num_eval_pts)
        d_errors[i, :] = errs
    end 
    cmap = cgrad(:viridis, scale=:exp)
    p1 = heatmap(xs, ds, d_errors, 
        xlabel="x'", ylabel="d", 
        title="$(enc.name) Median Uncertainty", 
        yticks=(6:20), xticks=(first(r):0.2:last(r)),
        c=cmap, clims=(0.0, 1.0), colorbar_title="Abs(Error)",
        )

    p2 = plot(xlabel="x'", ylabel="Abs(Error)", legend=:outertopright, title="All d slices")
    for (idx, d) in enumerate(ds)
        plot!(xs, d_errors[idx, :], label="d=$(d)")
    end
    p8 = plot(xs, d_errors[3, :], xlabel="x'", ylabel="Abs(Error)", label="d=8", legend=:outertopright)
    p12 = plot!(xs, d_errors[7, :], label="d=12")
    p16 = plot!(xs, d_errors[11, :], label="d=16")
    p20 = plot!(xs, d_errors[end, :], label="d=20")


    p = plot(p1, p2, p8, size=(1000, 500), bottom_margin=5mm, left_margin=5mm)
    
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


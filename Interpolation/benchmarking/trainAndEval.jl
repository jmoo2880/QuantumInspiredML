include("/Users/joshua/QuantumMay/QuantumInspiredML/LogLoss/RealRealHighDimension.jl");
include("/Users/joshua/QuantumMay/QuantumInspiredML/Interpolation/ForecastingMain.jl");

base_dir = cd("/Users/joshua/Documents/QuantumInspiredML/Sampling/benchmarking/nslvn")

train_file = "data/train_unscaled.jld2"
test_file = "data/test_unscaled.jld2"
output_folder = "/Users/joshua/QuantumMay/QuantumInspiredML/Interpolation/benchmarking/nslvn/fourier"
enc_basis = Basis("Fourier")

function generate_all_mps(train_file::String, test_file::String, output_folder::String, enc_basis::Basis)
    """Run benchmarks for a given basis"""
    train_f = jldopen(train_file, "r")
    test_f = jldopen(test_file, "r")
    X_train = read(train_f, "X_train")
    y_train = read(train_f, "y_train")
    X_test = read(test_f, "X_test")
    y_test = read(test_f, "y_test")
    X_val = X_test
    y_val = y_test
    close(train_f)
    close(test_f)

    setprecision(BigFloat, 128)
    Rdtype = Float64
    verbosity = 0
    d = 2:2:10
    chi = 10:5:30
    param_grid = collect(Iterators.product(d, chi))
    
    for (idx, (d_val, chi_val)) in enumerate(param_grid)

        # check to see whether params have already been computed
        fname = "$(lowercase(enc_basis.name))_d$(d_val)_chi$(chi_val).jld2"
        mps_file = joinpath(output_folder, fname)
        if isfile(mps_file)
            println("Skipping parameter combination: $idx: d = $d_val χ = $chi_val")
            continue
        end

        opts=Options(; nsweeps=15, chi_max=chi_val, update_iters=1, verbosity=verbosity, dtype=Complex{Rdtype}, lg_iter=KLD_iter,
        bbopt=BBOpt("CustomGD"), track_cost=false, eta=0.05, rescale = (false, true), d=d_val, aux_basis_dim=2, encoding=enc_basis)

        print_opts(opts)
        mps, info, train_states, test_states = fitMPS(X_train, y_train, X_val, y_val, X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=false)
        # save the trained mps
        fname = "$(lowercase(enc_basis.name))_d$(d_val)_chi$(chi_val).jld2"
        mps_file = joinpath(output_folder, fname)
        jldsave(mps_file; mps, info, opts)
    end

end

function evaluate_all_mps_forecast(dir_loc::String, data_loc::String, enc_basis::Basis, horizon::Int=50)
    """Function to evaluate the performance of each mps 
    in the directory.
    - dir_loc is the location of the directory containing all of the trained mps .jld2 files
    - dat_loc is the location of the scaled data in the appropriate range i.e., [-1, 1] or [0, 1]
    """
    # load all of the files in the directory  
    all_files = readdir(dir_loc)
    mps_files = filter(f -> endswith(f, ".jld2"), all_files)
    println("Found $(length(mps_files)) jld2 files.")
    scores = Dict()
    println("="^80)
    for f in mps_files
        fpath = joinpath(dir_loc, f)
        println("loaded: $f")
        fcast = unpack_class_states_and_samples(fpath, data_loc)
        scores_c0 = forecast_class_mode_analytic(fcast, 0, horizon, enc_basis)
        scores_c1 = forecast_class_mode_analytic(fcast, 1, horizon, enc_basis)
        scores_combined = vcat(scores_c0, scores_c1)
        dict_name = split(f, ".")[1]
        scores[dict_name] = scores_combined
        println("="^80)
        println("FINISHED $f")
        println("="^80)
    end
   
    return scores
    
end



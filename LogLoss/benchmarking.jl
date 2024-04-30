using JLD2

struct Result
    acc::Float64
    conf::Matrix{Float64}
    KLD::Float64
    MSE::Float64
end

function Result(d::Dict{String,Vector{Float64}},s::Dict{Symbol, Any})

    acc = d["test_acc"][end]
    conf = s[:confmat]
    KLD = d["test_KL_div"][end]
    MSE = d["test_loss"][end]
    return Result(acc, conf, KLD, MSE)
end

include("RealRealFast_generic.jl")

function save_status(path::String,chi::Int,d::Int,e::Encoding, chis::Vector{Int},ds::Vector{Int},encodings::Vector{Encoding})
    f = jldopen(path, "w")
    write(f, "chi", chi)
    write(f, "chis", chis)

    write(f, "d", d)
    write(f, "ds", ds)

    write(f, "e", e.name)
    write(f, "encodings", [enc.name for enc in encodings])
    close(f)
end

function check_status(path::String)
    f = jldopen(path, "r")
    chi = read(f, "chi")
    chis = read(f, "chis")

    d = read(f, "d")
    ds = read(f, "ds")

    e = Encoding(read(f, "e"))
    encodings = Encoding.(read(f, "encodings"))
    close(f)

    return chi, chis, d, ds, e, encodings
end


function check_status(path::String,chis::Vector{Int},ds::Vector{Int},encodings::Vector{Encoding})

    chi_r, chis_r, d_r, ds_r, e_r, encodings_r = check_status(path)

    if chis_r == chis && ds_r == ds && encodings_r == encodings
        println("Found interrupted benchmark, resuming")
        return true, [chi_r,d_r,e_r]

    else
        return false, [chis_r,ds_r,encodings_r]
    end
end

function logdata(fpath::String, W::MPS, info::Dict, train_states::timeSeriesIterable, test_states::timeSeriesIterable, opts::Options)
    f = open(fpath, "a")
    print_opts(opts; io=f)
    stats = get_training_summary(W, train_states, test_states; print_stats=true, io=f);

    sweep_summary(info; io=f)
    print(f, "\n\n/=======================================================================================================================================================\\ \n\n")
    close(f)
    return stats
end





bpath = "LogLoss/benchmarking/"

verbosity = 0
random_state=456
chi_init= 1

rescale = [false, true]
bbopt =BBOpt("CustomGD")
update_iters=1
eta=0.05
track_cost = false
lg_iter = KLD_iter


encodings = [Encoding("sahand"), Encoding("Fourier")] # Encoding.(["Stoudenmire", "Fourier", "Sahand", "Legendre"])


nsweeps = 2 # 10


chis = 10:5:15
ds = [2,3] #vcat(2:10,20,30)


output = Array{Result}(undef, length(encodings), length(ds), length(chis))


# checks
pstr = "$(random_state)_ns=$(nsweeps)_chis=$(chis)_ds=$(minimum(ds)):$(maximum(ds))"
chis = collect(chis)

path = bpath* pstr *"/"
svfol= path*"data/"
logpath = path # could change to a new folder if we wanted

logfile = logpath * "log_"* pstr * ".txt"
resfile = logpath * pstr * ".jld2"
statfile = logpath * pstr*"_status.jld2"


# resume if possible
esi=1
dsi=1
chi_si=1
if  isdir(path) && !isempty(readdir(path))
    if isfile(statfile)
        resume, upto = check_status(statfile, chis, ds, encodings)
        if resume
            esi= findfirst(upto[3], encodings)
            dsi= findfirst(upto[2], ds)
            chi_si= findfirst(upto[1], chis)
        else
            error("A status file exists but the parameters don't match!\nchis_r=$(upto[1])\nds=$(upto[2])\nencodings=$(upto[3])")
        end
    else
        while true
            print("A benchmark with these parameters already exists, continue? [y/n]: ")
            input = lowercase(readline())
            if input == "y"
                # the length is for safety so we can never recursively remove something terrible like "/" (shout out to the steam linux runtime)
                isdir(path) && length(path) >=3 && rm(path; recursive=true ) 
                break
            elseif input == "n"
                error("Aborting to conserve existing data")
            end
        end
    end
end

# make the folders if they dont already exist
!isdir(path) && mkdir(path)
!isdir(svfol) && mkdir(svfol)



#load training data 
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("LogLoss/datasets/ECG_train.txt", 
   "LogLoss/datasets/ECG_val.txt", "LogLoss/datasets/ECG_test.txt")
X_train = vcat(X_train, X_val)
y_train = vcat(y_train, y_val)

tstart=time()
for (ei,e) in enumerate(encodings[esi:end])
    e.iscomplex ? dtype = ComplexF64 : dtype = Float64
    for (di,d) in enumerate(ds[dsi:end])
        isodd(d) && titlecase(e.name) == "Sahand" && continue
        
        # generate the encodings
       # _, _, train_states, test_states = fitMPS(X_train, y_train, X_val, y_val, X_test, y_test; chi_init=1, opts=Options(;dtype=dtype, d=d, encoding=e), test_run=true)
        train_states = nothing
        test_states = nothing

        for (chi_i,chi_max) in enumerate(chis[chi_si:end])
            # save the status file 
            save_status(statfile, chi_max,d,e, chis,ds,encodings)

            opts=Options(; nsweeps=nsweeps, chi_max=chi_max, update_iters=update_iters, verbosity=verbosity, dtype=dtype, lg_iter=lg_iter,
                bbopt=bbopt, track_cost=track_cost, eta=eta, rescale = rescale, d=d, encoding=e)

            if isnothing(train_states) || isnothing(test_states) # ensures we only encode once per d
                W, info, train_states, test_states = fitMPS(X_train, y_train, X_val, y_val, X_test, y_test; random_state=random_state, chi_init=chi_init, opts=opts)
            else
                W, info, train_states, test_states = fitMPS(train_states, train_states, test_states; random_state=random_state, chi_init=chi_init, opts=opts)
            end
            stats = logdata(logfile, W, info, train_states, test_states, opts)


            println("Saving MPS, t=$(time()-tstart)")
            f = jldopen(svfol * "$(ei)_$(d)_$(chi_max).jld2", "w")
                write(f, "W", W)
                write(f, "info", info)
                # write(f, "train_states", train_states)
                # write(f, "test_states", test_states)
            close(f)
            output[ei,di,chi_i] = Result(info, stats)

            println("Saving Results, t=$(time()-tstart)")
            f = jldopen(resfile,"w")
                write(f, "output", output)
            close(f)
        end
    end

end
println(output)
# finished, remove the statfile
rm(statfile)
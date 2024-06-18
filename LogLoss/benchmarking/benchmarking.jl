using JLD2 # better for julia datastructures than hdf5

include("../RealRealHighDimension.jl")
include("benchUtils.jl")

bpath = "LogLoss/benchmarking/"

toydata = false
encode_classes_separately = false
train_classes_separately = false

verbosity = 0
random_state=456
chi_init= 4
tsgo = true

rescale = (false, true)

update_iters=1
eta=0.05
track_cost = false
lg_iter = KLD_iter


if tsgo
    bbopt = BBOpt("CustomGD", "TSGO") 
else
    bbopt = BBOpt("CustomGD")
end
# encodings = Basis.(["Fourier"])
# encodings = SplitBasis.([ "Hist Split Stoudenmire", "Hist Split Fourier", "Hist Split Legendre"])
encodings = [stoudenmire(), fourier(), legendre(), legendre(norm=false)]
#encodings = vcat(Basis("Stoudenmire"), Basis("Fourier"), Basis("Legendre"), SplitBasis.(["Hist Split Uniform", "Hist Split Stoudenmire", "Hist Split Fourier", "Hist Split Sahand", "Hist Split Legendre"]))


nsweeps = 25
chis = 10:5:25
ds = 2:4:14
aux_basis_dim=2
ramlimit = 451 # limits chimax * d to be less than this number, 451 corresponds to about 32GB of ram for a complex encoding


output = Array{Union{Result, Nothing}}(nothing, length(encodings), length(ds), length(chis))


# checks
vstring = train_classes_separately ? "Split_" : ""
tstring = tsgo ? "TSGO_" : ""
dstring = toydata ? "toy_" : ""
pstr = "v3_"*vstring*tstring * dstring * "eta$(eta)_$(random_state)_ns$(nsweeps)_chis$(chis)_ds$(minimum(ds)):$(maximum(ds))"

chis = collect(chis)
ds = collect(ds)

path = bpath* pstr *"/"
svfol= path*"data/"
logpath = path # could change to a new folder if we wanted

logfile = logpath * "log"* ".txt"
resfile = logpath  * "results.jld2"
statfile = logpath *"status.jld2"
finfile = logpath * "params.jld2"


# resume if possible

if  isdir(path) && !isempty(readdir(path))
    if isfile(statfile)
        resume, upto = check_status(statfile, chis, ds, encodings)
        if resume

            println("Found interrupted benchmark, resuming")
            println("(chi,d,e) = $upto")

            f = jldopen(resfile,"r")
            output = f["output"]
            close(f)

        else
            error("??? A status file exists but the parameters don't match!\nchis_r=$(upto[1])\nds=$(upto[2])\nencodings=$(upto[3])")
        end
    elseif isfile(finfile)
        while true
            print("A benchmark with these parameters already exists, overwrite the contents of \"$path\"? [y/n]: ")
            input = lowercase(readline())
            if input == "y"
                # the length is for safety so we can never recursively remove something terrible like "/" (shout out to the steam linux runtime)
                isdir(path) && length(path) >=3 && rm(path; recursive=true ) 
                break
            elseif input == "n"
                error("Aborting to conserve existing data")
            end
        end
    else
        error("A non benchmark folder with the name\n$path\nAlready exists")
    end

end

# make the folders and output file if they dont already exist
if !isdir(path) 
    mkdir(path)
    f = jldopen(resfile,"w")
    write(f, "output", output)
    close(f)
end

!isdir(svfol) && mkdir(svfol)



# load training data 
if !toydata
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("LogLoss/datasets/ECG_train.txt", 
    "LogLoss/datasets/ECG_val.txt", "LogLoss/datasets/ECG_test.txt")
    X_train = vcat(X_train, X_val)
    y_train = vcat(y_train, y_val)
else
    (X_train, y_train), (X_test, y_test) = generate_toy_timeseries(30, 100) 
    X_val = X_test
    y_val = y_test

end

tstart=time()
for (ei,e) in enumerate(encodings)
    dtype = e.iscomplex ? ComplexF64 : Float64

    for (di,d) in enumerate(ds)
        isodd(d) && titlecase(e.name) == "Sahand" && continue
        d != 2 && titlecase(e.name) == "Stoudenmire" && continue
        
        # generate the encodings
       # _, _, train_states, test_states = fitMPS(X_train, y_train, X_val, y_val, X_test, y_test; chi_init=1, opts=Options(;dtype=dtype, d=d, encoding=e), test_run=true)
        train_states_meta = nothing
        test_states_meta = nothing

        for (chi_i,chi_max) in enumerate(chis)
            !isnothing(output[ei,di,chi_i]) && continue # Resume where we left off
            if chi_max * d > ramlimit 
                # I only have 32GB of ram on my desktop
                println("Too Much RAM! skipping")
                continue
            end
        
            # save the status file 
            save_status(statfile, chi_max,d,e, chis, ds,encodings)

            opts=Options(; nsweeps=nsweeps, chi_max=chi_max, update_iters=update_iters, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
                bbopt=bbopt, track_cost=track_cost, eta=eta, rescale = rescale, d=d, aux_basis_dim=aux_basis_dim, encoding=e, encode_classes_separately=encode_classes_separately,
                train_classes_separately=train_classes_separately)

            print_opts(opts)

            # define these here so they escape the scope of the try block
            W, info = MPS(), Dict()

            try
                if isnothing(train_states_meta) || isnothing(test_states_meta) # ensures we only encode once per d
                    W, info, train_states_meta, test_states_meta = fitMPS(X_train, y_train, X_val, y_val, X_test, y_test; random_state=random_state, chi_init=chi_init, opts=opts)
                else
                    W, info, train_states_meta, test_states_meta = fitMPS(train_states_meta, train_states_meta, test_states_meta; random_state=random_state, chi_init=chi_init, opts=opts)
                end

            catch train_err
                if train_err isa ArgumentError && train_err.msg == "matrix contains Infs or NaNs"
                    @warn("SVD encountered infinite values, ignoring this set of parameters")
                    logdata(logfile, W, info, train_states_meta.timeseries, test_states_meta.timeseries, opts; err=true, err_str=train_err.msg)
                    continue
                else
                    throw(train_err)
                end
            end
            stats = logdata(logfile, W, info, train_states_meta.timeseries, test_states_meta.timeseries, opts; err=false)


            println("Saving MPS, t=$(time()-tstart)")

            svfile = svfol * "$(ei)_$(d)_$(chi_max).jld2"
            local f = jldopen(svfile, "w")
                write(f, "W", W)
                write(f, "info", info)
                # write(f, "train_states", train_states)
                # write(f, "test_states", test_states)
            close(f)
            save_status(svfile, chi_max,d,e, chis,ds,encodings, append=true)

            output[ei,di,chi_i] = Result(info, stats)

            println("Saving Results, t=$(time()-tstart)")
            local f = jldopen(resfile,"w")
                write(f, "output", output)
            close(f)
            save_status(resfile, chi_max,d,e, chis,ds, encodings, append=true)


        end
    end

end



f = open(logfile, "a")

print(f, "\n\n/=======================================================================================================================================================\\ \n\n")
print(f, "\n\n/=======================================================================================================================================================\\ \n\n")

tab_results(output, chis, ds, encodings; io=f, fancy_conf=true)
tab_results(output, chis, ds, encodings; io=f, conf=false)
close(f)

# finished, move the statfile
mv(statfile,finfile)

tab_results(output, chis, ds, encodings; fancy_conf=true)




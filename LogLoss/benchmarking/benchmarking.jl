using JLD2 # better for julia datastructures than hdf5

include("../RealRealFast_generic.jl")
include("benchUtils.jl")

bpath = "LogLoss/benchmarking/"

verbosity = 0
random_state=456
chi_init= 4

rescale = [false, true] 
bbopt =BBOpt("CustomGD")
update_iters=1
eta=0.05
track_cost = false
lg_iter = KLD_iter


encodings = Encoding.(["Stoudenmire", "Fourier", "Sahand", "Legendre"])


nsweeps = 20


chis = 5:5:50
ds = vcat(2:10,20,30)


output = Array{Union{Result, Nothing}}(nothing, length(encodings), length(ds), length(chis))


# checks
pstr = "$(random_state)_ns=$(nsweeps)_chis=$(chis)_ds=$(minimum(ds)):$(maximum(ds))"
chis = collect(chis)

path = bpath* pstr *"/"
svfol= path*"data/"
logpath = path # could change to a new folder if we wanted

logfile = logpath * "log"* ".txt"
resfile = logpath  * "results.jld2"
statfile = logpath *"status.jld2"
finfile = logpath * "params.jld2"


# resume if possible
esi=1
dsi=1
chi_si=1
if  isdir(path) && !isempty(readdir(path))
    if isfile(statfile)
        resume, upto = check_status(statfile, chis, ds, encodings)
        if resume
            esi= findfirst(e -> e == upto[3], encodings)
            dsi= findfirst(d -> d == upto[2], ds)
            chi_si= findfirst(chi -> chi == upto[1], chis)

            f = jldopen(resfile,"r")
            output = f["output"]
            close(f)

        else
            error("A status file exists but the parameters don't match!\nchis_r=$(upto[1])\nds=$(upto[2])\nencodings=$(upto[3])")
        end
    else
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
        d != 2 && titlecase(e.name) == "Stoudenmire" && continue
        
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

            svfile = svfol * "$(ei)_$(d)_$(chi_max).jld2"
            f = jldopen(svfile, "w")
                write(f, "W", W)
                write(f, "info", info)
                # write(f, "train_states", train_states)
                # write(f, "test_states", test_states)
            close(f)
            save_status(svfile, chi_max,d,e, chis,ds,encodings, append=true)

            output[ei,di,chi_i] = Result(info, stats)

            println("Saving Results, t=$(time()-tstart)")
            f = jldopen(resfile,"w")
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
close(f)

# finished, move the statfile
mv(statfile,finfile)

tab_results(output, chis, ds, encodings; fancy_conf=true)




import Base: /, *, ^, +, -, zero
import StatsBase: mean_and_std

struct Result
    acc::Float64
    acc_tr::Float64
    maxacc::Tuple{Float64, Integer}
    conf::Matrix{Float64}
    KLD::Float64
    minKLD::Tuple{Float64, Integer}
    KLD_tr::Float64
    MSE::Float64
    swtime::Float64
end

# julia crimes below -----------------------------------
for (fname) in [:/,:*,:-,:+,:^,]
    @eval begin
        function $fname(r::Result, d::Number)
            acc = $fname(r.acc, d)
            acc_tr = $fname(r.acc_tr, d)
            maxacc = ismissing(r.maxacc) ? missing : ($fname(r.maxacc[1], d), r.maxacc[2])
            conf = $fname.(r.conf, d)
            KLD = $fname(r.KLD, d)
            minKLD = ismissing(r.minKLD) ? missing : ($fname(r.minKLD[1], d), r.minKLD[2])
            KLD_tr = ismissing(r.KLD_tr) ? missing : $fname(r.KLD_tr,d)
            MSE = $fname(r.MSE, d)
            swtime = $fname(r.swtime, d)
    
            return Result(acc, acc_tr, maxacc, conf, KLD, minKLD, KLD_tr, MSE, swtime)
        end

        $fname(d::Number, r::Result) = $fname(r::Result, d::Number)

        function $fname(r1::Result, r2::Result)
            acc = $fname(r1.acc, r2.acc)
            acc_tr = $fname(r1.acc_tr, r2.acc_tr)
            maxacc = ismissing(r1.maxacc) ? missing : $fname.(r1.maxacc, r2.maxacc)
            conf = $fname.(r1.conf, r2.conf)
            KLD = $fname(r1.KLD, r2.KLD)
            minKLD = ismissing(r1.minKLD) ? missing : $fname.(r1.minKLD, r2.minKLD)
            KLD_tr = ismissing(r1.KLD_tr) ? missing : $fname(r1.KLD_tr,r2.KLD_tr)
            MSE = $fname(r1.MSE, r2.MSE)
            swtime = $fname(r1.swtime, r2.swtime)
    
            return Result(acc, acc_tr, maxacc, conf, KLD, minKLD, KLD_tr, MSE, swtime)
        end
    end
end

for (fname) in [:-,:(Base.abs), :(Base.abs2), :(Base.sqrt), :(Base.real), :(Base.conj), :(Base.imag)]
    @eval begin
        function $fname(r::Result)
            acc = $fname(r.acc)
            acc_tr = $fname(r.acc_tr)
            maxacc = ismissing(r.maxacc) ? missing : ($fname(r.maxacc[1]), r.maxacc[2])
            conf = $fname.(r.conf)
            KLD = $fname(r.KLD)
            minKLD = ismissing(r.minKLD) ? missing : ($fname(r.minKLD[1]), r.minKLD[2])
            KLD_tr = ismissing(r.KLD_tr) ? missing : $fname(r.KLD_tr)
            MSE = $fname(r.MSE)
            swtime = $fname(r.swtime)
    
            return Result(acc, acc_tr, maxacc, conf, KLD, minKLD, KLD_tr, MSE, swtime)
        end

    end
end

Base.length(::Result) = 1
Base.iterate(r::Result) = (r, nothing)
Base.iterate(::Result, ::Nothing) = nothing
Base.zero(::Result) = Result(0.,0.,(0., 0), zeros(2,2),0., (0., 0), 0., 0., 0.)
Base.zero(Result) = Result(0.,0., (0., 0), zeros(2,2),0., (0., 0), 0., 0., 0.)
Base.:+(::Result, y::Missing) = y

# dummy example for debugging
Result() = Result(1.,1., (1., 2), ones(2,2),1., (2., 4), 1., 1., 0.)

struct CompatResult
    acc::Float64
    maxacc::Union{Tuple{Float64, Integer}, Missing} # oopsie
    conf::Matrix{Float64}
    KLD::Float64
    minKLD::Union{Tuple{Float64, Integer}, Missing} # oopsie
    KLD_tr::Union{Float64,Missing} # oopsie
    MSE::Float64
end





function Result(d::Dict{String,Vector}, swi::Integer)

    acc = d["test_acc"][swi]
    acc_tr = d["train_acc"][swi]
    conf = d["test_conf"][swi]
    KLD = d["test_KL_div"][swi]
    KLD_tr = d["train_KL_div"][swi]
    MSE = d["test_loss"][swi]
    swtime = d["time_taken"][swi]

    maxacc = findmax(d["test_acc"])
    minKLD = findmin(d["test_KL_div"])
    return Result(acc, acc_tr, maxacc, conf, KLD, minKLD, KLD_tr,MSE,swtime)
end

function Result(d::Dict{String,Vector})
    return [Result(d, swi) for swi in 1:(length(d["test_acc"])-1)]
end

import Base: /, *, ^, +, -
import StatsBase: mean_and_std

struct Result
    acc::Float64
    maxacc::Tuple{Float64, Integer}
    conf::Matrix{Float64}
    KLD::Float64
    minKLD::Tuple{Float64, Integer}
    KLD_tr::Float64
    MSE::Float64
end

# julia crimes below -----------------------------------
for (fname) in [:/,:*,:-,:+,:^,]
    @eval begin
        function $fname(r::Result, d::Number)
            acc = $fname(r.acc, d)
            maxacc = ismissing(r.maxacc) ? missing : ($fname(r.maxacc[1], d), r.maxacc[2])
            conf = $fname.(r.conf, d)
            KLD = $fname(r.KLD, d)
            minKLD = ismissing(r.minKLD) ? missing : ($fname(r.minKLD[1], d), r.minKLD[2])
            KLD_tr = ismissing(r.KLD_tr) ? missing : $fname(r.KLD_tr,d)
            MSE = $fname(r.MSE, d)
    
            return Result(acc, maxacc, conf, KLD, minKLD, KLD_tr, MSE)
        end

        $fname(d::Number, r::Result) = $fname(r::Result, d::Number)

        function $fname(r1::Result, r2::Result)
            acc = $fname(r1.acc, r2.acc)
            maxacc = ismissing(r1.maxacc) ? missing : $fname.(r1.maxacc, r2.maxacc)
            conf = $fname.(r1.conf, r2.conf)
            KLD = $fname(r1.KLD, r2.KLD)
            minKLD = ismissing(r1.minKLD) ? missing : $fname.(r1.minKLD, r2.minKLD)
            KLD_tr = ismissing(r1.KLD_tr) ? missing : $fname(r1.KLD_tr,r2.KLD_tr)
            MSE = $fname(r1.MSE, r2.MSE)
    
            return Result(acc, maxacc, conf, KLD, minKLD, KLD_tr, MSE)
        end
    end
end

for (fname) in [:-,:(Base.abs), :(Base.abs2), :(Base.sqrt), :(Base.real), :(Base.conj), :(Base.imag)]
    @eval begin
        function $fname(r::Result)
            acc = $fname(r.acc)
            maxacc = ismissing(r.maxacc) ? missing : ($fname(r.maxacc[1]), r.maxacc[2])
            conf = $fname.(r.conf)
            KLD = $fname(r.KLD)
            minKLD = ismissing(r.minKLD) ? missing : ($fname(r.minKLD[1]), r.minKLD[2])
            KLD_tr = ismissing(r.KLD_tr) ? missing : $fname(r.KLD_tr)
            MSE = $fname(r.MSE)
    
            return Result(acc, maxacc, conf, KLD, minKLD, KLD_tr, MSE)
        end

    end
end

Base.length(::Result) = 1
Base.iterate(r::Result) = (r, nothing)
Base.iterate(::Result, ::Nothing) = nothing


struct CompatResult
    acc::Float64
    maxacc::Union{Tuple{Float64, Integer}, Missing} # oopsie
    conf::Matrix{Float64}
    KLD::Float64
    minKLD::Union{Tuple{Float64, Integer}, Missing} # oopsie
    KLD_tr::Union{Float64,Missing} # oopsie
    MSE::Float64
end

# dummy example for debugging
Result() = Result(1.,(1., 2), ones(2,2),1., (2., 4), 1., 1.)



function Result(d::Dict{String,Vector}, swi::Integer)

    acc = d["test_acc"][swi]
    conf = d["test_conf"][swi]
    KLD = d["test_KL_div"][swi]
    KLD_tr = d["train_KL_div"][swi]
    MSE = d["test_loss"][swi]

    maxacc = findmax(d["test_acc"])
    minKLD = findmin(d["test_KL_div"])
    return Result(acc, maxacc, conf, KLD, minKLD, KLD_tr,MSE)
end

function Result(d::Dict{String,Vector})
    return [Result(d, swi) for swi in 1:(length(d["test_acc"])-1)]
end
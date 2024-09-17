import Pkg
Pkg.activate(".")
using Distributed
using BenchmarkTools
using Random

# create processes
addprocs(31, exeflags="-t 1")
w = workers()

r = @spawnat w[1] (myid(), rand())
all_futures = [@spawnat i (myid(), rand()) for i in w]
@btime fetch.(all_futures)

# serial version
function sqrt_sum(A)
    s = zero(eltype(A))
    for i in eachindex(A)
        @inbounds s += sqrt(A[i])
    end
    return s
end

@btime sqrt_sum(100000)

# distributed version
@everywhere function sqrt_sum_range(A, r)
    s = zero(eltype(A))
    for i in r
        @inbounds s += sqrt(A[i])
    end
    return s
end

A = rand(100_000)
batch = Int(length(A) / 100)

@distributed (+) for r in [(1:batch) .+ offset for offset in 0:batch:length(A)-1]
    sqrt_sum_range(A, r)
end

@btime sum(pmap(r -> sqrt_sum_range(A, r), [(1:batch) .+ offset for offset in 0:batch:length(A)-1]))

@everywhere using Dagger

ctx = Context()

ctx.procs

Dagger.get_processors.(ctx.procs)

@everywhere function task()
    return (Distributed.myid(), Threads.threadid())
end

tasks = [Dagger.@spawn task() for _ in 1:10]
results = fetch.(tasks)
println("(Worker ID, Thread ID)")
println("Main process")
println(task())
println("Dagger tasks")
foreach(println, sort(results))

@everywhere function task_nested(a::Integer, b::Integer)
    return [Dagger.@spawn b+i for i in one(a):a]
end
rngs = [MersenneTwister(seed) for seed in 1:3]
a = Dagger.@spawn rand(rngs[1], 4:8)
b = Dagger.@spawn rand(rngs[2], 10:20)
c = Dagger.@spawn task_nested(fetch(a), fetch(b))
d = Dagger.@spawn rand(rngs[3], 10:20)
f = Dagger.@spawn mapreduce(fetch, +, fetch(c)) + fetch(d)
fetch(f)
using Distributed
addprocs(30)
@everywhere using Distributed
println("Number of workers: ", nworkers())
println("Total number of processes: ", nprocs())
@time result1 = @distributed (vcat) for i in 1:1000000
    [i, i^2, i^3]
end
println("Size of result1: ", size(result1))
@everywhere function estimate_pi(n)
    inside_circle = 0
    for i in 1:n
        x, y = rand(), rand()
        if x^2 + y^2 <= 1
            inside_circle += 1
        end
    end
    return 4 * inside_circle / n
end

@time result2 = @distributed (+) for i in 1:32
    estimate_pi(10000000)
end
println("Estimated value of pi (parallel): ", result2 / 32)


@time result2_sequential = estimate_pi(320000000)
rmprocs(workers())
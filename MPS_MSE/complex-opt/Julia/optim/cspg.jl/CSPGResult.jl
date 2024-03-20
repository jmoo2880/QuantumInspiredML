"""
    CSPGResult

Data structure that contains the results of the `cspg` optimization method. 

    `x`: Array containing final point (solution or best point found).
 
    `f`: Final objective function value.

    `gnorm`: Norm of the projected gradient on the bounds.

    `nit`: Number of iterations performed. 

    `nfeval`: Number of function evaluations.

    `ierr`: Status of execution: 

         0. Converged successfully to solution. 

         1. Maximum number of iterations achieved.

         2. Maximum number of function evaluations achieved.

    `return_from_callback`: Boolean indicating if the algorithm returned from the callback function.

"""
struct CSPGResult{T,F,G}
    x::T
    f::F
    gnorm::G
    nit::Int64
    nfeval::Int64
    ierr::Int64
    return_from_callback::Bool
end

function Base.show(io::IO, R::CSPGResult)
    println("")
    println(" CSPG RESULT: ")
    println("")
    if R.return_from_callback
        println(" Returned from callback function. ")
    else
        R.ierr == 0 && println(" Convergence achieved. ")
        R.ierr == 1 && println(" Maximum number of iterations (nitmax) reached.")
        R.ierr == 2 && println(" Maximum number of function evaluations (nfevalmax) reached.")
    end
    println("")
    println(" Final objective function value = ", R.f)
    print(" Sample of best point = $(typeof(R.x))[ ")
    for i = 1:min(length(R.x) - 1, 3)
        print("$(R.x[i]), ")
    end
    if length(R.x) > 4
        print("..., ")
    end
    println(R.x[length(R.x)], "]")
    println(" Projected gradient norm = ", R.gnorm)
    println("")
    println(" Number of iterations = ", R.nit)
    print(" Number of function evaluations = ", R.nfeval)
end

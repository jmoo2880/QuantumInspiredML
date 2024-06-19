# gridsearch hyperparameter opt

function hyperopt(encoding::Encoding, X_train::AbstractMatrix, y_train::AbstractVector, X_val::AbstractMatrix, y_val::AbstractVector; 
    method="GridSearch", 
    etas::Vector{Float64}, 
    sweepns::Vector{Integer}, 
    ds::Vector{Integer}, 
    chi_maxs::Vector{Integer}, 
    crossval::Bool=false,
    folds::Integer=1,
    kfoldseed::Integer=1234567, # overridded by the rng parameter
    rng::AbstractRNG=MersenneTwister(kfoldseed),
    update_iters::Integer=1,
    verbosity::Real=-1,
    dtype::Type = encoding.iscomplex ? ComplexF64 : Float64,
    loss_grad::Function=loss_grad_KLD,
    bbopt::BBOpt=BBOpt("CustomGD", "TSGO"),
    track_cost::Bool=false,
    rescale::Tuple{Bool,Bool}=(false, true),
    aux_basis_dim::Integer=2,
    encode_classes_separately::Bool=false,
    train_classes_separately::Bool=false
    )



    Xs = [X_train ; X_val]
    ys = [y_train ; y_val]

    if cross_val
        throw(ErrorException("crossval Not Implemented"))
        ntrs = length(y_train)
        nvals = length(y_val)

        fold_inds = Array{Vector{Integer}}(undef, folds, 2)


        fold_inds[1,:] = [collect(1:ntrs), collect((ntrs+1):(ntrs+nvals))]

        for i in 2:folds
            inds = randperm(rng, ntrs + nvals)
            fold_inds[i,:] = [inds[1:ntrs], inds[(ntrs+1):(ntrs+nvals)]]
        end
    else
        fold_inds = [collect(1:ntrs), collect((ntrs+1):(ntrs+nvals))]
    end

    # do encodings for all d and all folds

    # do Ws for all d

    results = Array{Union{Result,Missing}}(missing, length(etas), length(sweepns), length(ds), length(chi_maxs), length())
    #TODO threadsfor crossvalidate, or ThreadPools.@qthreads because the loads aren't balanced
    # threadsfor eta since it doesn't affect sweep time
    for d in ds, chi_max in chi_maxs
        opts=Options(; nsweeps=nsweeps, chi_max=chi_max, update_iters=update_iters, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad,
                bbopt=bbopt, track_cost=track_cost, eta=eta, rescale = rescale, d=d, aux_basis_dim=aux_basis_dim, encoding=encoding, encode_classes_separately=encode_classes_separately,
                train_classes_separately=train_classes_separately)

    end
end
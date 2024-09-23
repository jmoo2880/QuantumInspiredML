#### Import these two libraries first and in this order!!
using GenericLinearAlgebra
using MKL
####
import MLJModelInterface
using MLJ
const MMI = MLJModelInterface

include("../LogLoss/RealRealHighDimension.jl")


MMI.@mlj_model mutable struct MPSClassifier <: MMI.Deterministic
    reformat_verbosity::Int=-1 # The verbosity used when reformatting/encoding data
    #### contents of MPSOptions goes here
    verbosity::Int=1::(_>0) #### Unused by MLJ 
    nsweeps::Int=5::(_>0) # Number of MPS optimisation sweeps to perform (Both forwards and Backwards)
    chi_max::Int=15::(_>0) # Maximum bond dimension allowed within the MPS during the SVD step
    eta::Float64=0.01::(_>0) # The gradient descent step size for CustomGD. For Optim and OptimKit this serves as the initial step size guess input into the linesearch
    d::Int=2::(_>0) # The dimension of the feature map or "Encoding". This is the true maximum dimension of the feature vectors. For a splitting encoding, d = num_splits * aux_basis_dim
    encoding::Symbol=:Legendre_No_Norm::(_ in [:Legendre, :Legendre_No_Norm, :Stoudenmire, :Fourier, :Sahand]) # The type of encoding to use, see structs.jl and encodings.jl for the various options. Can be just a time (in)dependent orthonormal basis, or a time (in)dependent basis mapped onto a number of "splits" which distribute tighter basis functions where the sites of a timeseries are more likely to be measured.  
    projected_basis::Bool=false # whether to pass project=true to the basis
    aux_basis_dim::Int=2::(_>0) # (NOT IMPLEMENTED) If encoding::SplitBasis, serves as the auxilliary dimension of a basis mapped onto the split encoding, so that num_bins = d / aux_basis_dim. Unused if encoding::Basis
    cutoff::Float64=1E-10::(_>0) # Size based cutoff for the number of singular values in the SVD (See Itensors SVD documentation)
    update_iters::Int=1::(_>0) # Maximum number of optimiser iterations to perform for each bond tensor optimisation. E.G. The number of steps of (Conjugate) Gradient Descent used by CustomGD, Optim or OptimKit
    dtype::DataType=(model_encoding(encoding).iscomplex ? ComplexF64 : Float64)::(_<:Number) # The datatype of the elements of the MPS as well as the encodings. Set to a complex value only if necessary for the encoding type. Supports the arbitrary precsion types BigFloat and Complex{BigFloat}
    loss_grad::Symbol=:KLD::(_ in [:KLD, :MSE, :Mixed]) # The type of cost function to use for training the MPS, typically Mean Squared Error or KL Divergence. Must return a vector or pair [cost, dC/dB]
    bbopt::Symbol=:TSGO::(_ in [:GD, :TSGO, :Optim, :OptimKit]) # Which Black Box optimiser to use, options are Optim or OptimKit derived solvers which work well for MSE costs, or CustomGD, which is a standard gradient descent algorithm with fixed stepsize which seems to give the best results for KLD cost 
    track_cost::Bool=false # Whether to print the cost at each Bond tensor site to the terminal while training, mostly useful for debugging new cost functions or optimisers
    rescale::Tuple{Bool,Bool}=(false, true) # Has the form rescale = (before::Bool, after::Bool) and tells the optimisor where to enforce the normalisation of the MPS during training, either calling normalise!(BT) before or after BT is updated. Note that for an MPS that starts in canonical form, rescale = (true,true) will train identically to rescale = (false, true) but may be less performant.
    train_classes_separately::Bool=false # whether the the trainer takes the average MPS loss over all classes or whether it considers each class as a separate problem
    encode_classes_separately::Bool=false # only relevant for a histogram splitbasis. If true, then the histogram used to determine the bin widths for encoding class A is composed of only data from class A, etc. Functionally, this causes the encoding method to vary depending on the class
    return_encoding_meta_info::Bool=false # Debug flag: Whether to return the normalised data as well as the histogram bins for the splitbasis types
    minmax::Bool=true # Whether to apply a minmax norm to the encoded data after it's been SigmoidTransformed
    exit_early::Bool=true # whether to stop training when train_acc = 1
    sigmoid_transform::Bool=true # Whether to apply a sigmoid transform to the data before minmaxing
    init_rng::Int=1234::( _ > 0) # SEED ONLY IMPLEMENTED (Itensors fault) random number generator or seed 
    chi_init::Int=4::(_>0) # Initial bond dimension of the randomMPS
    log_level::Int=0 # 0 for nothing, >0 to save losses, accs, and conf mat. #TODO implement finer grain control
end

include("MLJUtils.jl")

function MMI.fit(m::MPSClassifier, verbosity::Int, X, y, decode)
    opts = MPSOptions(m; verbosity=verbosity)
    _,_,enc_opt = Options(MPSOptions(m; verbosity=m.reformat_verbosity))

    if enc_opt.train_classes_separately || enc_opt.encode_classes_separately
        # ensure both at once, makes hyperparam optimisation easier
        enc_opt = _set_options(enc_opt, train_classes_separately=true, encode_classes_separately=true)
        opts = _set_options(opts, train_classes_separately=true, encode_classes_separately=true )
    end

    # ensure dtype can accomodate encoding (again, for assisting hyperopts)
    natural_dtype = enc_opt.encoding.iscomplex ? ComplexF64 : Float64
    enc_opt = _set_options(enc_opt, dtype=promote_type(enc_opt.dtype, natural_dtype))
    opts = _set_options(opts, dtype=promote_type(enc_opt.dtype, natural_dtype))


    X_enc = encoderows(enc_opt, X, y) # I would love to use MMI.reformat for this but I the sigmoid transform depends on the choice of samples in selectrows and that isn't called for predict

    W, info, _, _ = fitMPS(X_enc, EncodedTimeseriesSet(); opts=opts)
    cache = nothing
    report = (info = info,)
    return ((decode, enc_opt, W), cache, report)
end

function MMI.predict(::MPSClassifier, fitresult, Xnew)
    decode, enc_opt, W = fitresult
    Xnew_enc = encoderows(get_siteinds(W), enc_opt, Xnew)

    yhat = MPSpredict(W, Xnew_enc)
    return decode.(yhat)
end

# for fit:
MMI.reformat(::MPSClassifier, X, y) = (MMI.matrix(X; transpose=true), MMI.int(y), MMI.decoder(y))
MMI.selectrows(::MPSClassifier, I, Xmatrix, y, meta...) = (view(Xmatrix, :, I), view(y, I), meta...)

# for predict:
MMI.reformat(::MPSClassifier, X) = (MMI.matrix(X; transpose=true),)
MMI.selectrows(::MPSClassifier, I, Xmatrix) = (view(Xmatrix, :, I), )


include("interpolation_hyperopt_hack.jl")
# # for fit:
# MMI.reformat(::MPSClassifier, X, y) =
#     (MMI.matrix(X; transpose=true), MMI.int(y), classes(y))

# function MMI.selectrows(m::MPSClassifier, I, Xmatrix, y, meta...) 
#     yv = view(y, I)
#     opts = Options(m)
#     X_enc = encoderows(opts, view(Xmatrix, :, I), yv)
#     (X_enc, yv, meta...)

# end

# # for predict:
# MMI.reformat(::MPSClassifier, X) = (MMI.matrix(X; transpose=true),)

# function MMI.selectrows(m::MPSClassifier, I, Xmatrix)
#     opts = Options(m)
#     X_enc = encoderows(opts, view(Xmatrix, :, I))
#     return (X_enc,)
# end
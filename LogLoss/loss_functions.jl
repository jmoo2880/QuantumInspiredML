abstract type LossFunction <: Function end

abstract type KLDLoss <: LossFunction end
abstract type MSELoss <: LossFunction end

struct Loss_Grad_MSE <: MSELoss end
struct Loss_Grad_KLD <: KLDLoss end
struct Loss_Grad_KLD_slow <: KLDLoss end

struct Loss_Grad_mixed <: LossFunction end
struct Loss_Grad_default <: LossFunction end


loss_grad_MSE = Loss_Grad_MSE()
loss_grad_KLD = Loss_Grad_KLD()
loss_grad_KLD_slow = Loss_Grad_KLD_slow()

loss_grad_mixed = Loss_Grad_mixed()
loss_grad_default = Loss_Grad_default()

#######################################################


function yhat_phitilde(BT::ITensor, LEP::PCacheCol, REP::PCacheCol, 
    product_state::PState, lid::Int, rid::Int)
    """Return yhat and phi_tilde for a bond tensor and a single product state"""
    ps= product_state.pstate
    phi_tilde = conj(ps[lid] * ps[rid]) # phi tilde 


    if lid == 1
        if rid !== length(ps) # the fact that we didn't notice the previous version breaking for a two site MPS for nearly 5 months is hilarious
            # at the first site, no LE
            # formatted from left to right, so env - product state, product state - env
            phi_tilde *=  REP[rid+1]
        end
       
    elseif rid == length(ps)
        # terminal site, no RE
        phi_tilde *= LEP[lid-1] 
    else
        # we are in the bulk, both LE and RE exist
        phi_tilde *= LEP[lid-1] * REP[rid+1]

    end


    yhat = BT * phi_tilde # NOT a complex inner product !! 

    return yhat, phi_tilde

end

function yhat_phitilde!(phi_tilde::ITensor, BT::ITensor, LEP::PCacheCol, REP::PCacheCol, 
    product_state::PState, lid::Int, rid::Int)
    """Return yhat and phi_tilde for a bond tensor and a single product state"""
    ps= product_state.pstate
    phi_tilde .= conj(ps[lid] * ps[rid]) # phi tilde 


    if lid == 1
        if rid !== length(ps) # the fact that we didn't notice the previous version breaking for a two site MPS for nearly 5 months is hilarious
            # at the first site, no LE
            # formatted from left to right, so env - product state, product state - env
            phi_tilde *=  REP[rid+1]
        end
       
    elseif rid == length(ps)
        # terminal site, no RE
        phi_tilde *= LEP[lid-1] 
    else
        # we are in the bulk, both LE and RE exist
        phi_tilde *= LEP[lid-1] * REP[rid+1]

    end


    yhat = BT * phi_tilde # NOT a complex inner product !! 

    return yhat

end





#####################################################################################################  MSE LOSS

function MSE_iter(BT_c::ITensor, LEP::PCacheCol, REP::PCacheCol,
    product_state::PState, lid::Int, rid::Int) 
    """Computes the Mean squared error loss function derived from KL divergence and its gradient"""


    yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

    # convert the label to ITensor
    label_idx = inds(yhat)[1]
    y = onehot(label_idx => (product_state.label + 1))

    diff_sq = abs2.(yhat - y)
    sum_of_sq_diff = sum(diff_sq)
    loss = 0.5 * real(sum_of_sq_diff)

    # construct the gradient - return dC/dB
    gradient = (yhat - y) * conj(phi_tilde)

    return [loss, gradient]

end


function (::Loss_Grad_MSE)(::TrainSeparate{false}, BT::ITensor, LE::PCache, RE::PCache,
    ETSs::EncodedTimeseriesSet, lid::Int, rid::Int)
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    TSs = ETSs.timeseries
    loss,grad = mapreduce((LEP,REP, prod_state) -> MSE_iter(BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE), eachcol(RE),TSs)
    
    loss /= length(TSs)
    grad ./= length(TSs)

    return loss, grad

end


################################################################################################### KLD loss


function KLD_iter(BT_c::ITensor, LEP::PCacheCol, REP::PCacheCol,
    product_state::PState, lid::Int, rid::Int) 
    """Computes the complex valued logarithmic loss function derived from KL divergence and its gradient"""
    
    # it is assumed that BT has no label index, so yhat is a rank 0 tensor
    yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

    f_ln = yhat[1]
    loss = -log(abs2(f_ln))

    # construct the gradient - return dC/dB
    gradient = -conj(phi_tilde / f_ln) 

    return [loss, gradient]

end

function KLD_iter!(phit_scaled::ITensor, BT_c::ITensor, LEP::PCacheCol, REP::PCacheCol,
    product_state::PState, lid::Int, rid::Int) 
    """Computes the complex valued logarithmic loss function derived from KL divergence and its gradient"""
    
    # it is assumed that BT has no label index, so yhat is a rank 0 tensor
    yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

    f_ln = yhat[1]
    loss = -log(abs2(f_ln))

    # construct the gradient - return dC/dB
    gradient = -conj(phi_tilde / f_ln) 

    return [loss, gradient]

end


function (::Loss_Grad_KLD)(::TrainSeparate{true}, BT::ITensor, LE::PCache, RE::PCache,
    ETSs::EncodedTimeseriesSet, lid::Int, rid::Int)
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    cnums = ETSs.class_distribution
    TSs = ETSs.timeseries
    label_idx = findindex(BT, "f(x)")

    losses = zero(real(eltype(BT))) # ITensor(real(eltype(BT)), label_idx)
    grads = ITensor(eltype(BT), inds(BT))

    i_prev = 0
    for (ci, cn) in enumerate(cnums)
        y = onehot(label_idx => ci)
        bt = BT * y

        c_inds = (i_prev+1):(cn + i_prev)
        loss, grad = mapreduce((LEP,REP, prod_state) -> KLD_iter(bt,LEP,REP,prod_state,lid,rid),+, eachcol(view(LE, :, c_inds)), eachcol(view(RE, :, c_inds)),TSs[c_inds])

        losses += loss / cn # * y # maybe doing this with a combiner instead will be more efficient
        grads += grad * y / cn
        i_prev += cn
    end


    return losses, grads

end

function (::Loss_Grad_KLD)(::TrainSeparate{false}, BT::ITensor, LE::PCache, RE::PCache,
    ETSs::EncodedTimeseriesSet, lid::Int, rid::Int)
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    cnums = ETSs.class_distribution
    TSs = ETSs.timeseries
    label_idx = findindex(BT, "f(x)")

    losses = zero(real(eltype(BT)))
    grads = ITensor(eltype(BT), inds(BT))

 
    i_prev=0
    for (ci, cn) in enumerate(cnums)
        y = onehot(label_idx => ci)
        bt = BT * y

        c_inds = (i_prev+1):(cn+i_prev)
        loss, grad = mapreduce((LEP,REP, prod_state) -> KLD_iter(bt,LEP,REP,prod_state,lid,rid),+, eachcol(view(LE, :, c_inds)), eachcol(view(RE, :, c_inds)),TSs[c_inds])
        losses += loss # maybe doing this with a combiner instead will be more efficient
        grads .+= grad * y 
        #### equivalent without mapreduce
        # for ci in c_inds 
        #     # mapreduce((LEP,REP, prod_state) -> KLD_iter(bt,LEP,REP,prod_state,lid,rid),+, eachcol(view(LE, :, c_inds)), eachcol(view(RE, :, c_inds)),TSs[c_inds])
        #     # valid = map(ts -> ts.label_index == ci, TSs[c_inds]) |> all
        #     LEP = view(LE, :, ci)
        #     REP = view(RE, :, ci)
        #     prod_state = TSs[ci]
        #     loss, grad = KLD_iter(bt,LEP,REP,prod_state,lid,rid)

        #     losses += loss # maybe doing this with a combiner instead will be more efficient
        #     grads .+= grad * y 
        # end
        #####
        
        i_prev += cn
    end

    losses /= length(TSs)
    grads ./= length(TSs)


    return losses, grads

end

###################################################################################################  Mixed loss


function mixed_iter(BT_c::ITensor, LEP::PCacheCol, REP::PCacheCol,
    product_state::PState, lid::Int, rid::Int; alpha=5) 
    """Returns the loss and gradient that results from mixing the logarithmic loss and mean squared error loss with mixing parameter alpha"""

    yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

    # convert the label to ITensor
    label_idx = inds(yhat)[1]
    y = onehot(label_idx => (product_state.label + 1))
    f_ln = (yhat *y)[1]
    log_loss = -log(abs2(f_ln))

    # construct the gradient - return dC/dB
    log_gradient = -y * conj(phi_tilde / f_ln) # mult by y to account for delta_l^lambda

    # MSE
    diff_sq = abs2.(yhat - y)
    sum_of_sq_diff = sum(diff_sq)
    MSE_loss = 0.5 * real(sum_of_sq_diff)

    # construct the gradient - return dC/dB
    MSE_gradient = (yhat - y) * conj(phi_tilde)


    return [log_loss + alpha*MSE_loss, log_gradient + alpha*MSE_gradient]

end


function (::Loss_Grad_mixed)(::TrainSeparate{false}, BT::ITensor, LE::PCache, RE::PCache,
    ETSs::EncodedTimeseriesSet, lid::Int, rid::Int; alpha=5)
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    TSs = ETSs.timeseries
    loss,grad = mapreduce((LEP,REP, prod_state) -> mixed_iter(BT,LEP,REP,prod_state,lid,rid; alpha=alpha),+, eachcol(LE), eachcol(RE),TSs)
    
    loss /= length(TSs)
    grad ./= length(TSs)

    return loss, grad

end


######################### old  generic Loss_Grad function
function (::Loss_Grad_default)(::TrainSeparate{false}, BT::ITensor, LE::PCache, RE::PCache,
    ETSs::EncodedTimeseriesSet, lid::Int, rid::Int; lg_iter::Function=KLD_iter)
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    TSs = ETSs.timeseries
    loss,grad = mapreduce((LEP,REP, prod_state) -> lg_iter(BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE), eachcol(RE),TSs)
    
    loss /= length(TSs)
    grad ./= length(TSs)

    return loss, grad

end

function (::Loss_Grad_default)(::TrainSeparate{true}, BT::ITensor, LE::PCache, RE::PCache,
    ETSs::EncodedTimeseriesSet, lid::Int, rid::Int)
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    cnums = ETSs.class_distribution
    TSs = ETSs.timeseries
    label_idx = findindex(BT, "f(x)")

    losses = ITensor(real(eltype(BT)), label_idx)
    grads = ITensor(eltype(BT), inds(BT))

    i_prev=0
    for (ci, cn) in enumerate(cnums)
        y = onehot(label_idx => ci)

        c_inds = (i_prev+1):cn
        loss, grad = mapreduce((LEP,REP, prod_state) -> KLD_iter(BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE)[c_inds], eachcol(RE)[c_inds],TSs[c_inds])

        losses += loss  / cn # maybe doing this with a combiner instead will be more efficient
        grads += grad / cn
        i_prev = cn
    end


    return losses, grads

end


#################### Do not use, for reproducing old data only
function KLD_iter_slow(BT_c::ITensor, LEP::PCacheCol, REP::PCacheCol,
    product_state::PState, lid::Int, rid::Int) 
    """Computes the complex valued logarithmic loss function derived from KL divergence and its gradient"""


    yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

    # convert the label to ITensor
    label_idx = inds(yhat)[1]
    y = onehot(label_idx => (product_state.label + 1))
    f_ln = (yhat *y)[1]
    loss = -log(abs2(f_ln))

    # construct the gradient - return dC/dB
    gradient = -y * conj(phi_tilde / f_ln) # mult by y to account for delta_l^lambda

    return [loss, gradient]

end

function (::Loss_Grad_KLD_slow)(::TrainSeparate{false}, BT::ITensor, LE::PCache, RE::PCache,
    ETSs::EncodedTimeseriesSet, lid::Int, rid::Int)
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    TSs = ETSs.timeseries
    l = findindex(BT, "f(x)")
    loss,grad = Folds.mapreduce((LEP,REP, prod_state) -> [1.0, onehot(l => prod_state.label_index)] .* KLD_iter(BT * onehot(l => prod_state.label_index),LEP,REP,prod_state,lid,rid),+, eachcol(LE), eachcol(RE),TSs)
    
    loss /= length(TSs)
    grad ./= length(TSs)

    return loss, grad

end






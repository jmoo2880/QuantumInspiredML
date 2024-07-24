function OBC(W::MPS,
    training_states_meta::EncodedTimeseriesSet,
    testing_states_meta::EncodedTimeseriesSet,
    training_information::Dict;
    opts::Options=Options(),
    loss_grads::AbstractArray,
    bbopts::AbstractArray)

    sites = siteinds(W)
    verbosity = opts.verbosity
    training_states = training_states_meta.timeseries
    testing_states = testing_states_meta.timeseries
    tsep = TrainSeparate{opts.train_classes_separately}() # value type to determine training style
    nsweeps = opts.nsweeps

    for itS = 1:nsweeps
        start = time()
        verbosity > -1 && println("Using optimiser $(bbopts[itS].name) with the \"$(bbopts[itS].fl)\" algorithm")
        verbosity > -1 && println("Starting backward sweeep: [$itS/$nsweeps]")

        LE, RE = construct_caches(W, training_states, length(W); going_left=true)

        for j = (length(sites)-1):-1:1
            #print("Bond $j")
            # j tracks the LEFT site in the bond tensor (irrespective of sweep direction)
            BT = W[j] * W[(j+1)] # create bond tensor
            BT_new = apply_update(tsep, BT, LE, RE, j, (j+1), training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                    dtype=opts.dtype, loss_grad=loss_grads[itS], bbopt=bbopts[itS],
                                    track_cost=opts.track_cost, eta=opts.eta, rescale = opts.rescale) # optimise bond tensor


            left_site_indices = inds(W[j])
            label_index = findindex(BT, "f(x)")
            left_site_indices = [left_site_indices..., label_index]

            # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
            lsn, rsn = decomposeBT(BT_new, j, (j+1), left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=true, dtype=opts.dtype)
                
            # update the caches to reflect the new tensors
            update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=true)
            # place the updated sites back into the MPS
            W[j] = lsn
            W[(j+1)] = rsn
        end
    
        # add time taken for backward sweep.
        verbosity > -1 && println("Backward sweep finished.")
        
        # finished a full backward sweep, reset the caches and start again
        # this can be simplified dramatically, only need to reset the LE
        LE, RE = construct_caches(W, training_states, 1; going_left=false)
        
        verbosity > -1 && println("Starting forward sweep: [$itS/$nsweeps]")

        for j = 1:(length(sites)-1)
            #print("Bond $j")
            BT = W[j] * W[(j+1)]
            BT_new = apply_update(tsep, BT, LE, RE, j, (j+1), training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                    dtype=opts.dtype, loss_grad=loss_grads[itS], bbopt=bbopts[itS],
                                    track_cost=opts.track_cost, eta=opts.eta, rescale=opts.rescale) # optimise bond tensor


            left_site_indices = [inds(W[j])...]
            label_index = findindex(BT, "f(x)")
            deleteat!(left_site_indices, findfirst(left_site_indices .== label_index))

            lsn, rsn = decomposeBT(BT_new, j, (j+1), left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=false, dtype=opts.dtype)
            update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=false)
            W[j] = lsn
            W[(j+1)] = rsn
        end

        
        finish = time()

        time_elapsed = finish - start
        
        # add time taken for full sweep 
        verbosity > -1 && println("Finished sweep $itS.")

        # compute the loss and acc on both training and validation sets
        train_loss, train_acc = MSE_loss_acc(W, training_states)
        test_loss, test_acc, conf = MSE_loss_acc_conf(W, testing_states)
        train_KL_div = KL_div(W, training_states)
        test_KL_div = KL_div(W, testing_states)

        # dot_errs = test_dot(W, testing_states)

        # if !isempty(dot_errs)
        #     @warn "Found mismatching values between inner() and MPS_contract at Sites: $dot_errs"
        # end
        verbosity > -1 && println("Training MSE loss: $train_loss | Training acc. $train_acc." )
        verbosity > -1 && println("Testing MSE loss: $test_loss | Testing acc. $test_acc." )
        verbosity > -1 && println("")
        verbosity > -1 && println("Training KL Divergence: $train_KL_div.")
        verbosity > -1 && println("Test KL Divergence: $test_KL_div.")
        verbosity > -1 && println("Test conf: $conf.")


        push!(training_information["train_loss"], train_loss)
        push!(training_information["train_acc"], train_acc)
        push!(training_information["test_loss"], test_loss)
        push!(training_information["test_acc"], test_acc)
        push!(training_information["time_taken"], time_elapsed)
        push!(training_information["train_KL_div"], train_KL_div)
        push!(training_information["test_KL_div"], test_KL_div)
        push!(training_information["test_conf"], conf)
    end
    return W, training_information
end

function PBC_left(W::MPS,
    training_states_meta::EncodedTimeseriesSet,
    testing_states_meta::EncodedTimeseriesSet,
    training_information::Dict;
    opts::Options=Options(),
    loss_grads::AbstractArray,
    bbopts::AbstractArray)

    sites = siteinds(W)
    verbosity = opts.verbosity
    training_states = training_states_meta.timeseries
    testing_states = testing_states_meta.timeseries
    loss_grad_terminal = (args...) -> loss_grad_KLD(W, args...)
    nsweeps = opts.nsweeps
    tsep = TrainSeparate{opts.train_classes_separately}() # value type to determine training style

    for itS = 1:nsweeps
        start = time()
        verbosity > -1 && println("Using optimiser $(bbopts[itS].name) with the \"$(bbopts[itS].fl)\" algorithm")
        verbosity > -1 && println("Starting left sweeep: [$itS/$nsweeps]")

        LE, RE = construct_caches(W, training_states, length(W); going_left=true)

        for j = (length(sites)-1):-1:1
            #print("Bond $j")
            # j tracks the LEFT site in the bond tensor (irrespective of sweep direction)
            BT = W[j] * W[(j+1)] # create bond tensor
            BT_new = apply_update(tsep, BT, LE, RE, j, (j+1), training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                    dtype=opts.dtype, loss_grad=loss_grads[itS], bbopt=bbopts[itS],
                                    track_cost=opts.track_cost, eta=opts.eta, rescale = opts.rescale) # optimise bond tensor

            left_site_indices = inds(W[j])
            label_index = findindex(BT, "f(x)")
            left_site_indices = [left_site_indices..., label_index]

            # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
            lsn, rsn = decomposeBT(BT_new, j, (j+1), left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=true, dtype=opts.dtype)
                
            # update the caches to reflect the new tensors
            update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=true)
            # place the updated sites back into the MPS
            W[j] = lsn
            W[(j+1)] = rsn
        end
        
        # optimise over terminal ends of MPS
        lid = length(sites)
        rid = 1
        BT = W[lid] * W[rid]
        BT_new = apply_update(tsep, BT, LE, RE, lid, rid, training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                    dtype=opts.dtype, loss_grad=loss_grad_terminal, bbopt=bbopts[itS],
                                    track_cost=opts.track_cost, eta=opts.eta, rescale = opts.rescale)
        
        left_site_indices = inds(W[lid])
        label_index = findindex(BT, "f(x)")
        left_site_indices = [left_site_indices..., label_index]

        # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
        lsn, rsn = decomposeBT(BT_new, lid, rid, left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=true, dtype=opts.dtype)
        W[lid] = lsn
        W[rid] = rsn
        # add time taken for backward sweep.
        verbosity > -1 && println("Left sweep finished.")
        
        finish = time()

        time_elapsed = finish - start
        
        # add time taken for full sweep 
        verbosity > -1 && println("Finished sweep $itS.")

        # compute the loss and acc on both training and validation sets
        train_loss, train_acc = MSE_loss_acc(W, training_states)
        test_loss, test_acc, conf = MSE_loss_acc_conf(W, testing_states)
        train_KL_div = KL_div(W, training_states)
        test_KL_div = KL_div(W, testing_states)

        # dot_errs = test_dot(W, testing_states)

        # if !isempty(dot_errs)
        #     @warn "Found mismatching values between inner() and MPS_contract at Sites: $dot_errs"
        # end
        verbosity > -1 && println("Training MSE loss: $train_loss | Training acc. $train_acc." )
        verbosity > -1 && println("Testing MSE loss: $test_loss | Testing acc. $test_acc." )
        verbosity > -1 && println("")
        verbosity > -1 && println("Training KL Divergence: $train_KL_div.")
        verbosity > -1 && println("Test KL Divergence: $test_KL_div.")
        verbosity > -1 && println("Test conf: $conf.")
        

        push!(training_information["train_loss"], train_loss)
        push!(training_information["train_acc"], train_acc)
        push!(training_information["test_loss"], test_loss)
        push!(training_information["test_acc"], test_acc)
        push!(training_information["time_taken"], time_elapsed)
        push!(training_information["train_KL_div"], train_KL_div)
        push!(training_information["test_KL_div"], test_KL_div)
        push!(training_information["test_conf"], conf)
    end
    return W, training_information
end

function PBC_right(W::MPS,
    training_states_meta::EncodedTimeseriesSet,
    testing_states_meta::EncodedTimeseriesSet,
    training_information::Dict;
    opts::Options=Options(),
    loss_grads::AbstractArray,
    bbopts::AbstractArray)

    sites = siteinds(W)
    verbosity = opts.verbosity
    training_states = training_states_meta.timeseries
    testing_states = testing_states_meta.timeseries
    loss_grad_terminal = (args...) -> loss_grad_KLD(W, args...)
    nsweeps = opts.nsweeps
    tsep = TrainSeparate{opts.train_classes_separately}() # value type to determine training style

    for itS = 1:nsweeps
        start = time()
        verbosity > -1 && println("Using optimiser $(bbopts[itS].name) with the \"$(bbopts[itS].fl)\" algorithm")
        verbosity > -1 && println("Starting right sweeep: [$itS/$nsweeps]")
        
        
        
        # optimise over terminal ends of MPS
        lid = length(sites)
        rid = 1
        BT = W[lid] * W[rid]
        LE, RE = construct_caches(W, training_states, 1; going_left=false)
        BT_new = apply_update(tsep, BT, LE, RE, lid, rid, training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                    dtype=opts.dtype, loss_grad=loss_grad_terminal, bbopt=bbopts[itS],
                                    track_cost=opts.track_cost, eta=opts.eta, rescale = opts.rescale)
        
        left_site_indices = [inds(W[lid])...]
        label_index = findindex(BT, "f(x)")
        deleteat!(left_site_indices, findfirst(left_site_indices .== label_index))

        # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
        lsn, rsn = decomposeBT(BT_new, lid, rid, left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=false, dtype=opts.dtype)
        W[lid] = lsn
        W[rid] = rsn
        LE, RE = construct_caches(W, training_states, 1; going_left=false)
        for j = 1:(length(sites)-1)
            #print("Bond $j")
            BT = W[j] * W[(j+1)]
            BT_new = apply_update(tsep, BT, LE, RE, j, (j+1), training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                    dtype=opts.dtype, loss_grad=loss_grads[itS], bbopt=bbopts[itS],
                                    track_cost=opts.track_cost, eta=opts.eta, rescale=opts.rescale) # optimise bond tensor


            left_site_indices = [inds(W[j])...]
            label_index = findindex(BT, "f(x)")
            deleteat!(left_site_indices, findfirst(left_site_indices .== label_index))

            lsn, rsn = decomposeBT(BT_new, j, (j+1), left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=false, dtype=opts.dtype)
            update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=false)
            W[j] = lsn
            W[(j+1)] = rsn
        end
        
        
        # add time taken for backward sweep.
        verbosity > -1 && println("Right sweep finished.")
        
        finish = time()

        time_elapsed = finish - start
        
        # add time taken for full sweep 
        verbosity > -1 && println("Finished sweep $itS.")

        # compute the loss and acc on both training and validation sets
        train_loss, train_acc = MSE_loss_acc(W, training_states)
        test_loss, test_acc, conf = MSE_loss_acc_conf(W, testing_states)
        train_KL_div = KL_div(W, training_states)
        test_KL_div = KL_div(W, testing_states)

        # dot_errs = test_dot(W, testing_states)

        # if !isempty(dot_errs)
        #     @warn "Found mismatching values between inner() and MPS_contract at Sites: $dot_errs"
        # end
        verbosity > -1 && println("Training MSE loss: $train_loss | Training acc. $train_acc." )
        verbosity > -1 && println("Testing MSE loss: $test_loss | Testing acc. $test_acc." )
        verbosity > -1 && println("")
        verbosity > -1 && println("Training KL Divergence: $train_KL_div.")
        verbosity > -1 && println("Test KL Divergence: $test_KL_div.")
        verbosity > -1 && println("Test conf: $conf.")
        

        push!(training_information["train_loss"], train_loss)
        push!(training_information["train_acc"], train_acc)
        push!(training_information["test_loss"], test_loss)
        push!(training_information["test_acc"], test_acc)
        push!(training_information["time_taken"], time_elapsed)
        push!(training_information["train_KL_div"], train_KL_div)
        push!(training_information["test_KL_div"], test_KL_div)
        push!(training_information["test_conf"], conf)
    end
    return W, training_information
end

function PBC_both_two(W::MPS,
    training_states_meta::EncodedTimeseriesSet,
    testing_states_meta::EncodedTimeseriesSet,
    training_information::Dict;
    opts::Options=Options(),
    loss_grads::AbstractArray,
    bbopts::AbstractArray)

    test_lists = []
    sites = siteinds(W)
    verbosity = opts.verbosity
    training_states = training_states_meta.timeseries
    testing_states = testing_states_meta.timeseries
    loss_grad_terminal = (args...) -> loss_grad_KLD(W, args...)
    nsweeps = opts.nsweeps
    tsep = TrainSeparate{opts.train_classes_separately}() # value type to determine training style

    for itS = 1:nsweeps
        test_list = []
        start = time()
        verbosity > -1 && println("Using optimiser $(bbopts[itS].name) with the \"$(bbopts[itS].fl)\" algorithm")
        if itS % 4 == 1 || itS % 4 == 2
            verbosity > -1 && println("Starting left sweeep: [$itS/$nsweeps]")
            
            LE, RE = construct_caches(W, training_states, length(W); going_left=true)
            push!(test_list, find_label(W)[1])
            for j = (length(sites)-1):-1:1
                #print("Bond $j")
                # j tracks the LEFT site in the bond tensor (irrespective of sweep direction)
                BT = W[j] * W[(j+1)] # create bond tensor
                BT_new = apply_update(tsep, BT, LE, RE, j, (j+1), training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                        dtype=opts.dtype, loss_grad=loss_grads[itS], bbopt=bbopts[itS],
                                        track_cost=opts.track_cost, eta=opts.eta, rescale = opts.rescale) # optimise bond tensor

                left_site_indices = inds(W[j])
                label_index = findindex(BT, "f(x)")
                left_site_indices = [left_site_indices..., label_index]

                # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
                lsn, rsn = decomposeBT(BT_new, j, (j+1), left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=true, dtype=opts.dtype)
                    
                # update the caches to reflect the new tensors
                update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=true)
                # place the updated sites back into the MPS
                W[j] = lsn
                W[(j+1)] = rsn
                push!(test_list, find_label(W)[1])
            end
            
            # optimise over terminal ends of MPS
            lid = length(sites)
            rid = 1
            BT = W[lid] * W[rid]
            BT_new = apply_update(tsep, BT, LE, RE, lid, rid, training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                        dtype=opts.dtype, loss_grad=loss_grad_terminal, bbopt=bbopts[itS],
                                        track_cost=opts.track_cost, eta=opts.eta, rescale = opts.rescale)
            
            left_site_indices = inds(W[lid])
            label_index = findindex(BT, "f(x)")
            left_site_indices = [left_site_indices..., label_index]

            # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
            lsn, rsn = decomposeBT(BT_new, lid, rid, left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=true, dtype=opts.dtype)
            W[lid] = lsn
            W[rid] = rsn
            # add time taken for backward sweep.
            verbosity > -1 && println("Left sweep finished.")
        elseif itS % 4 == 3 || itS % 4 == 0
            verbosity > -1 && println("Starting right sweeep: [$itS/$nsweeps]")
            
            # optimise over terminal ends of MPS
            lid = length(sites)
            rid = 1
            BT = W[lid] * W[rid]
            LE, RE = construct_caches(W, training_states, 1; going_left=false)
            BT_new = apply_update(tsep, BT, LE, RE, lid, rid, training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                        dtype=opts.dtype, loss_grad=loss_grad_terminal, bbopt=bbopts[itS],
                                        track_cost=opts.track_cost, eta=opts.eta, rescale = opts.rescale)
            
            left_site_indices = [inds(W[lid])...]
            label_index = findindex(BT, "f(x)")
            deleteat!(left_site_indices, findfirst(left_site_indices .== label_index))

            # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
            lsn, rsn = decomposeBT(BT_new, lid, rid, left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=false, dtype=opts.dtype)
            push!(test_list, find_label(W)[1])
            W[lid] = lsn
            W[rid] = rsn
            push!(test_list, find_label(W)[1])
            LE, RE = construct_caches(W, training_states, 1; going_left=false)
            for j = 1:(length(sites)-1)
                #print("Bond $j")
                BT = W[j] * W[(j+1)]
                BT_new = apply_update(tsep, BT, LE, RE, j, (j+1), training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                        dtype=opts.dtype, loss_grad=loss_grads[itS], bbopt=bbopts[itS],
                                        track_cost=opts.track_cost, eta=opts.eta, rescale=opts.rescale) # optimise bond tensor


                left_site_indices = [inds(W[j])...]
                label_index = findindex(BT, "f(x)")
                deleteat!(left_site_indices, findfirst(left_site_indices .== label_index))

                lsn, rsn = decomposeBT(BT_new, j, (j+1), left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=false, dtype=opts.dtype)
                update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=false)
                W[j] = lsn
                W[(j+1)] = rsn
                push!(test_list, find_label(W)[1])
            end
            verbosity > -1 && println("Right sweep finished.")
        end    
        # add time taken for backward sweep.
        finish = time()

        time_elapsed = finish - start
        
        # add time taken for full sweep 
        verbosity > -1 && println("Finished sweep $itS.")

        # compute the loss and acc on both training and validation sets
        train_loss, train_acc = MSE_loss_acc(W, training_states)
        test_loss, test_acc, conf = MSE_loss_acc_conf(W, testing_states)
        train_KL_div = KL_div(W, training_states)
        test_KL_div = KL_div(W, testing_states)

        # dot_errs = test_dot(W, testing_states)

        # if !isempty(dot_errs)
        #     @warn "Found mismatching values between inner() and MPS_contract at Sites: $dot_errs"
        # end
        verbosity > -1 && println("Training MSE loss: $train_loss | Training acc. $train_acc." )
        verbosity > -1 && println("Testing MSE loss: $test_loss | Testing acc. $test_acc." )
        verbosity > -1 && println("")
        verbosity > -1 && println("Training KL Divergence: $train_KL_div.")
        verbosity > -1 && println("Test KL Divergence: $test_KL_div.")
        verbosity > -1 && println("Test conf: $conf.")
        

        push!(training_information["train_loss"], train_loss)
        push!(training_information["train_acc"], train_acc)
        push!(training_information["test_loss"], test_loss)
        push!(training_information["test_acc"], test_acc)
        push!(training_information["time_taken"], time_elapsed)
        push!(training_information["train_KL_div"], train_KL_div)
        push!(training_information["test_KL_div"], test_KL_div)
        push!(training_information["test_conf"], conf)
        push!(test_lists, test_list)
    end
    return W, training_information, test_lists
end

function PBC_both(W::MPS,
    training_states_meta::EncodedTimeseriesSet,
    testing_states_meta::EncodedTimeseriesSet,
    training_information::Dict;
    opts::Options=Options(),
    loss_grads::AbstractArray,
    bbopts::AbstractArray)

    test_lists = []
    sites = siteinds(W)
    verbosity = opts.verbosity
    training_states = training_states_meta.timeseries
    testing_states = testing_states_meta.timeseries
    loss_grad_terminal = (args...) -> loss_grad_KLD(W, args...)
    nsweeps = opts.nsweeps
    tsep = TrainSeparate{opts.train_classes_separately}() # value type to determine training style

    for itS = 1:nsweeps
        test_list = []
        start = time()
        verbosity > -1 && println("Using optimiser $(bbopts[itS].name) with the \"$(bbopts[itS].fl)\" algorithm")

        if itS % 2 == 1
            verbosity > -1 && println("Starting left sweeep: [$itS/$nsweeps]")
            
            LE, RE = construct_caches(W, training_states, length(W); going_left=true)
            #println(LE[1])
            push!(test_list, find_label(W)[1])
            for j = (length(sites)-1):-1:1
                #print("Bond $j")
                # j tracks the LEFT site in the bond tensor (irrespective of sweep direction)
                BT = W[j] * W[(j+1)] # create bond tensor
                BT_new = apply_update(tsep, BT, LE, RE, j, (j+1), training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                        dtype=opts.dtype, loss_grad=loss_grads[itS], bbopt=bbopts[itS],
                                        track_cost=opts.track_cost, eta=opts.eta, rescale = opts.rescale) # optimise bond tensor

                left_site_indices = inds(W[j])
                label_index = findindex(BT, "f(x)")
                left_site_indices = [left_site_indices..., label_index]

                # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
                lsn, rsn = decomposeBT(BT_new, j, (j+1), left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=true, dtype=opts.dtype)
                    
                # update the caches to reflect the new tensors
                update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=true)
                # place the updated sites back into the MPS
                W[j] = lsn
                W[(j+1)] = rsn
                push!(test_list, find_label(W)[1])
            end
            
            # optimise over terminal ends of MPS
            lid = length(sites)
            rid = 1
            BT = W[lid] * W[rid]
            BT_new = apply_update(tsep, BT, LE, RE, lid, rid, training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                        dtype=opts.dtype, loss_grad=loss_grad_terminal, bbopt=bbopts[itS],
                                        track_cost=opts.track_cost, eta=opts.eta, rescale = opts.rescale)
            
            left_site_indices = inds(W[lid])
            label_index = findindex(BT, "f(x)")
            left_site_indices = [left_site_indices..., label_index]

            # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
            lsn, rsn = decomposeBT(BT_new, lid, rid, left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=true, dtype=opts.dtype)
            W[lid] = lsn
            W[rid] = rsn
            # add time taken for backward sweep.
            verbosity > -1 && println("Left sweep finished.")
        elseif itS % 2 == 0
            verbosity > -1 && println("Starting right sweeep: [$itS/$nsweeps]")
            
            # optimise over terminal ends of MPS
            lid = length(sites)
            rid = 1
            BT = W[lid] * W[rid]
            LE, RE = construct_caches(W, training_states, 1; going_left=false)
            BT_new = apply_update(tsep, BT, LE, RE, lid, rid, training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                        dtype=opts.dtype, loss_grad=loss_grad_terminal, bbopt=bbopts[itS],
                                        track_cost=opts.track_cost, eta=opts.eta, rescale = opts.rescale)
            
            left_site_indices = [inds(W[lid])...]
            label_index = findindex(BT, "f(x)")
            deleteat!(left_site_indices, findfirst(left_site_indices .== label_index))

            # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
            lsn, rsn = decomposeBT(BT_new, lid, rid, left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=false, dtype=opts.dtype)
            push!(test_list, find_label(W)[1])
            W[lid] = lsn
            W[rid] = rsn
            push!(test_list, find_label(W)[1])
            LE, RE = construct_caches(W, training_states, 1; going_left=false)
            for j = 1:(length(sites)-1)
                #print("Bond $j")
                BT = W[j] * W[(j+1)]
                BT_new = apply_update(tsep, BT, LE, RE, j, (j+1), training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                        dtype=opts.dtype, loss_grad=loss_grads[itS], bbopt=bbopts[itS],
                                        track_cost=opts.track_cost, eta=opts.eta, rescale=opts.rescale) # optimise bond tensor


                left_site_indices = [inds(W[j])...]
                label_index = findindex(BT, "f(x)")
                deleteat!(left_site_indices, findfirst(left_site_indices .== label_index))

                lsn, rsn = decomposeBT(BT_new, j, (j+1), left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=false, dtype=opts.dtype)
                update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=false)
                W[j] = lsn
                W[(j+1)] = rsn
                push!(test_list, find_label(W)[1])
            end
            verbosity > -1 && println("Right sweep finished.")
        end    
        # add time taken for backward sweep.
        finish = time()

        time_elapsed = finish - start
        
        # add time taken for full sweep 
        verbosity > -1 && println("Finished sweep $itS.")

        # compute the loss and acc on both training and validation sets
        train_loss, train_acc = MSE_loss_acc(W, training_states)
        test_loss, test_acc, conf = MSE_loss_acc_conf(W, testing_states)
        train_KL_div = KL_div(W, training_states)
        test_KL_div = KL_div(W, testing_states)

        # dot_errs = test_dot(W, testing_states)

        # if !isempty(dot_errs)
        #     @warn "Found mismatching values between inner() and MPS_contract at Sites: $dot_errs"
        # end
        verbosity > -1 && println("Training MSE loss: $train_loss | Training acc. $train_acc." )
        verbosity > -1 && println("Testing MSE loss: $test_loss | Testing acc. $test_acc." )
        verbosity > -1 && println("")
        verbosity > -1 && println("Training KL Divergence: $train_KL_div.")
        verbosity > -1 && println("Test KL Divergence: $test_KL_div.")
        verbosity > -1 && println("Test conf: $conf.")
        

        push!(training_information["train_loss"], train_loss)
        push!(training_information["train_acc"], train_acc)
        push!(training_information["test_loss"], test_loss)
        push!(training_information["test_acc"], test_acc)
        push!(training_information["time_taken"], time_elapsed)
        push!(training_information["train_KL_div"], train_KL_div)
        push!(training_information["test_KL_div"], test_KL_div)
        push!(training_information["test_conf"], conf)
        push!(test_lists, test_list)
    end
    return W, training_information, test_lists
end

function optimise_bond(W::MPS,
    lid::Int,
    rid::Int,
    LE::PCache,
    RE::PCache,
    itS::Int,
    training_states_meta::EncodedTimeseriesSet,
    testing_states_meta::EncodedTimeseriesSet;
    opts::Options=Options(),
    loss_grads::AbstractArray,
    bbopts::AbstractArray,
    going_left::Bool)
    #looks at left_id and figures out, based on direction we're going, if we need to do terminal optimisation or usual optimisation

    test_lists = []
    verbosity = opts.verbosity
    training_states = training_states_meta.timeseries
    testing_states = testing_states_meta.timeseries
    loss_grad_terminal = (args...) -> loss_grad_KLD(W, args...)
    tsep = TrainSeparate{opts.train_classes_separately}() # value type to determine training style
    terminal = false
    if lid == length(W)
        terminal = true
    end

    if going_left && terminal
        #terminal optimisation here
        BT = W[lid] * W[rid]
        BT_new = apply_update(tsep, BT, LE, RE, lid, rid, training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                    dtype=opts.dtype, loss_grad=loss_grad_terminal, bbopt=bbopts[itS],
                                    track_cost=opts.track_cost, eta=opts.eta, rescale = opts.rescale)
        left_site_indices = inds(W[lid])
        label_index = findindex(BT, "f(x)")
        left_site_indices = [left_site_indices..., label_index]

        # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
        lsn, rsn = decomposeBT(BT_new, lid, rid, left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=true, dtype=opts.dtype)
        W[lid] = lsn
        W[rid] = rsn
    elseif going_left && !terminal
    #non terminal optimisation here
        BT = W[lid] * W[(rid)] # create bond tensor
        BT_new = apply_update(tsep, BT, LE, RE, lid, rid, training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                dtype=opts.dtype, loss_grad=loss_grads[itS], bbopt=bbopts[itS],
                                track_cost=opts.track_cost, eta=opts.eta, rescale = opts.rescale) # optimise bond tensor

        left_site_indices = inds(W[lid])
        label_index = findindex(BT, "f(x)")
        left_site_indices = [left_site_indices..., label_index]

        # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
        lsn, rsn = decomposeBT(BT_new, lid, rid, left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=true, dtype=opts.dtype)
            
        # update the caches to reflect the new tensors
        update_caches!(lsn, rsn, LE, RE, lid, rid, training_states; going_left=true)
        # place the updated sites back into the MPS
        W[lid] = lsn
        W[rid] = rsn
    elseif !going_left && terminal
        #terminal optiomisation here
        # optimise over terminal ends of MPS
        println("hi")
        BT = W[lid] * W[rid]
        BT_new = apply_update(tsep, BT, LE, RE, lid, rid, training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                    dtype=opts.dtype, loss_grad=loss_grad_terminal, bbopt=bbopts[itS],
                                    track_cost=opts.track_cost, eta=opts.eta, rescale = opts.rescale)
        
        left_site_indices = [inds(W[lid])...]
        label_index = findindex(BT, "f(x)")
        deleteat!(left_site_indices, findfirst(left_site_indices .== label_index))

        # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
        lsn, rsn = decomposeBT(BT_new, lid, rid, left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=false, dtype=opts.dtype)
        W[lid] = lsn
        W[rid] = rsn
        #LE, RE = construct_caches(W, training_states; going_left=false)
    elseif !going_left && !terminal
        #non terminal optimisation here
        BT = W[lid] * W[rid]
        BT_new = apply_update(tsep, BT, LE, RE, lid, rid, training_states_meta; iters=opts.update_iters, verbosity=verbosity, 
                                dtype=opts.dtype, loss_grad=loss_grads[itS], bbopt=bbopts[itS],
                                track_cost=opts.track_cost, eta=opts.eta, rescale=opts.rescale) # optimise bond tensor


        left_site_indices = [inds(W[lid])...]
        label_index = findindex(BT, "f(x)")
        deleteat!(left_site_indices, findfirst(left_site_indices .== label_index))

        lsn, rsn = decomposeBT(BT_new, lid, rid, left_site_indices; chi_max=opts.chi_max, cutoff=opts.cutoff, going_left=false, dtype=opts.dtype)
        update_caches!(lsn, rsn, LE, RE, lid, rid, training_states; going_left=false)
        W[lid] = lsn
        W[rid] = rsn
    end
    return W, LE, RE
end

function PBC_random(W::MPS,
    training_states_meta::EncodedTimeseriesSet,
    testing_states_meta::EncodedTimeseriesSet,
    training_information::Dict;
    opts::Options=Options(),
    loss_grads::AbstractArray,
    bbopts::AbstractArray)
    
    test_lists = []
    label_id = find_label(W)[1]
    nsweeps = opts.nsweeps
    training_states = training_states_meta.timeseries
    testing_states = testing_states_meta.timeseries
    rng = MersenneTwister(opts.random_walk_seed)
    for itS = 1:nsweeps
        test_list = []
        push!(test_list, find_label(W)[1])
        start = time()
        verbosity > -1 && println("Using optimiser $(bbopts[itS].name) with the \"$(bbopts[itS].fl)\" algorithm")
        extra_sweep = ceil(Int, rand(rng) * length(W))
        no_bond_tensors = length(W) + extra_sweep
        if itS % 2 == 1 #left
            verbosity > -1 && println("Starting left sweeep: [$itS/$nsweeps]")
            LE, RE = construct_caches(W, training_states, label_id; going_left=true)
            final_point = label_id - no_bond_tensors
            for j = label_id-1:-1:final_point
                lid = mod(j, length(W))
                rid = lid + 1
                if lid == 0
                    lid = length(W)
                end
                W, LE, RE = optimise_bond(W, lid, rid, LE, RE, itS, training_states_meta, testing_states_meta, 
                opts=opts, loss_grads=loss_grads, bbopts=bbopts, going_left = true)
                push!(test_list, find_label(W)[1])
                if lid == length(W)
                    LE, RE = construct_caches(W, training_states, length(W); going_left=true)
                end
            end
            verbosity > -1 && println("Left sweep finished.")
            label_id = find_label(W)[1]
        elseif itS % 2 == 0 #right
            verbosity > -1 && println("Starting right sweeep: [$itS/$nsweeps]")
            LE, RE = construct_caches(W, training_states, label_id; going_left=false)
            final_point = label_id + no_bond_tensors
            for j = label_id:1:final_point-1
                lid = mod(j, length(W))
                rid = lid + 1
                if lid == 0
                    lid = length(W)
                end
                W, LE, RE = optimise_bond(W, lid, rid, LE, RE, itS, training_states_meta, testing_states_meta, 
                opts=opts, loss_grads=loss_grads, bbopts=bbopts, going_left = false)
                push!(test_list, find_label(W)[1])
                if lid == length(W)
                    LE, RE = construct_caches(W, training_states, 1; going_left=false)
                end
            end
            verbosity > -1 && println("Right sweep finished.")
            label_id = find_label(W)[1]
        end
        finish = time()

        time_elapsed = finish - start
        
        # add time taken for full sweep 
        verbosity > -1 && println("Finished sweep $itS.")

        # compute the loss and acc on both training and validation sets
        train_loss, train_acc = MSE_loss_acc(W, training_states)
        test_loss, test_acc, conf = MSE_loss_acc_conf(W, testing_states)
        train_KL_div = KL_div(W, training_states)
        test_KL_div = KL_div(W, testing_states)

        # dot_errs = test_dot(W, testing_states)

        # if !isempty(dot_errs)
        #     @warn "Found mismatching values between inner() and MPS_contract at Sites: $dot_errs"
        # end
        verbosity > -1 && println("Training MSE loss: $train_loss | Training acc. $train_acc." )
        verbosity > -1 && println("Testing MSE loss: $test_loss | Testing acc. $test_acc." )
        verbosity > -1 && println("")
        verbosity > -1 && println("Training KL Divergence: $train_KL_div.")
        verbosity > -1 && println("Test KL Divergence: $test_KL_div.")
        verbosity > -1 && println("Test conf: $conf.")
        

        push!(training_information["train_loss"], train_loss)
        push!(training_information["train_acc"], train_acc)
        push!(training_information["test_loss"], test_loss)
        push!(training_information["test_acc"], test_acc)
        push!(training_information["time_taken"], time_elapsed)
        push!(training_information["train_KL_div"], train_KL_div)
        push!(training_information["test_KL_div"], test_KL_div)
        push!(training_information["test_conf"], conf)
        push!(test_lists, test_list)
    end
    return W, training_information, test_lists
end
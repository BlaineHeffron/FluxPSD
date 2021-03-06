#=
TrainPSD:
- Julia version: 
- Author: blaine
- Date: 2020-07-06
=#
using Flux
using Flux: logitcrossentropy
using Base.Iterators: partition
using Printf, BSON
using Parameters: @with_kw
using CUDA
if has_cuda()
    try
        import CuArrays
        CuArrays.allowscalar(false)
    catch ex
        @warn "CUDA is installed, but CuArrays.jl fails to load" exception=(ex,catch_backtrace())
    end
end
include("CommonFunctions.jl")

@with_kw mutable struct Args
    lr::Float64 = 3e-3
    epochs::Int = 100
    batch_size = 1000 #number of events per batch for each type
    savepath::String = "./model"
    n_train_evts::Int = 100000 #total number of events used for training for each type
    n_test_evts::Int = 20000 #total number of events used for testing for each type
    n_samples::Int = 150
    nx::Int = 14
    ny::Int = 11
end

function train(; kws...)
    args,ntype,modelname,train_dataset,test_dataset,indirs = init(; kws...)
    @info("Loading data set")
    sptrain, sptest = getData(args,indirs,ntype,train_dataset,test_dataset)

    @info("Building basic CNN model...")
    model = buildBasicCNN(args,ntype)

    # Load model and datasets onto GPU, if enabled
    sptrain = gpu.(sptrain)
    sptest = gpu.(sptest)
    model = gpu(model)

    # write the dataset to file
    writeToFile(train_dataset,joinpath(args.savepath,string(modelname,"_train_files.txt")))
    writeToFile(test_dataset,joinpath(args.savepath,string(modelname,"_test_files.txt")))

    # Make sure our model is nicely precompiled before starting our training loop
    model(sptrain[1][1])

    function loss(x, y)
        ŷ = model(x)
        return logitcrossentropy(ŷ, y)
    end

    # Train our model with the given training set using the ADAM optimizer and
    # printing out performance against the test set as we go.
    opt = ADAM(args.lr)

    @info("Beginning training loop...")
    best_acc = 0.0
    last_improvement = 0
    for epoch_idx in 1:args.epochs
        # Train for a single epoch
        Flux.train!(loss, params(model), sptrain, opt)

        # Terminate on NaN
        if anynan(paramvec(model))
            @error "NaN params"
            break
        end

        # Calculate accuracy:
        acc = accuracy(sptest..., model,ntype)

        @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
        # If our accuracy is good enough, quit out.
        if acc >= 0.999
            @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc >= best_acc
            @info(string(" -> New best accuracy! Saving model out to ",modelname,".bson"))
            BSON.@save joinpath(args.savepath,string(modelname,".bson")) params=cpu.(params(model)) epoch_idx acc
            best_acc = acc
            last_improvement = epoch_idx
        end

        # If we haven't seen improvement in 5 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
            opt.eta /= 10.0
            @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end

        if epoch_idx - last_improvement >= 10
            @warn(" -> We're calling this converged.")
            break
        end
    end
end



cd(@__DIR__)
@time train()
@time test()

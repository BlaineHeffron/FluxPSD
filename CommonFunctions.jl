include("Dataset.jl")
using HDF5;
using SparseArrays: sparse
using Base.Iterators: partition
using Flux: onehotbatch, onecold, logitcrossentropy
using Statistics: mean

function getDetCoord(n::Int32,num_x::Int)
    #n = 2*seg + i (where i = 0 or 1 left or right)
    #nx = seg%ncol, ny = floor(seg/ncol)
    nc::Int = n%2
    seg::Int = floor((n-nc)/2)
    nx::Int = seg%num_x + 1 #julia indices start at 1
    ny::Int = floor(seg/num_x) + 1 
    return nx,ny,nc
end
function getName(indirs)
    modelname = ""
    i = 0
    for d in indirs
        if !isdir(d)
            error("Error: argument " + d + " is not a directory")
        else
            if i > 0
                modelname = string(modelname,"_" , last(splitdir(d)))
            else
                modelname = string(modelname, last(splitdir(d)))
            end
        end
        i+=1
    end
    return modelname
end

function fillDataArrays(x::Array{UInt16,4},y::Array{UInt8,1},indirs,filelist::Dataset,n_evts_per_type::Int64,nx::Int64,excludeflist::Dataset,nsamp)
    evtcounter = 0
    n = 1
    i = 0
    for d in indirs
        evtcounter = readDir(d,x,evtcounter,filelist,n_evts_per_type,nx,excludeflist,nsamp)
        #simplistic assignment of 0,1,2,3 etc for different particle types
        while n <= evtcounter
            y[n] = i
            n+=1
        end
        i+=1
    end
end


function readDir(inputdir::String,train_x::Array{UInt16,4},n::Int64,fs::Dataset,maxevts::Int64,nx,fileexclude::Dataset,nsamp)
    #reads all WaveformSim files into Int16 2d array
    nEvts = 0
    for (root, dirs, files) in walkdir(inputdir)
        for file in files
            if endswith(file,"WaveformSim.h5")
                nm = joinpath(root,file)
                if hasFile(fileexclude,nm)
                    continue
                end
                thisevts = readHDF(nm,train_x,n,maxevts-nEvts,nx,nsamp)
                nEvts += thisevts
                n += thisevts
                addFile(fs,nm,(0,thisevts-1))
                if nEvts >= maxevts
                    return n
                end
            end
        end
    end
    return n
end

function readHDF(fname::String,dmx::Array{UInt16,4},offset,maxevts,numx,nsamp)
    nevents = 0
    c = h5open(fname, "r") do fid
        data = read(fid,"Waveforms")
        #println("the string: \t", typeof(data),"\t",data)
        curevt = -1
        for i in data
            if i.evt != curevt
                curevt = i.evt
                nevents += 1
                if nevents > maxevts
                    nevents -= 1
                    return  #returns to the end of the h5open block
                end
            end
            n = 1
            nx,ny,nc = getDetCoord(i.det,numx)
            for x in i.waveform
                #mapping the 2 pmt's samples to the "channels" for the convolution
                dmx[nx,ny,nc*nsamp+n,offset+nevents] = x
                n+=1
            end
            #@show i.waveform
        end
    end
    return nevents
end


function makeMinibatch(X, Y, idxs,ntypes,nx,ny,nsamp)
    X_batch = Array{UInt16}(undef, nx,ny,nsamp*2, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = X[:,:,:,idxs[i]]
    end
    Y_batch = onehotbatch(Y[idxs], 0:(ntypes-1))
    return (X_batch, Y_batch)
end

paramvec(m) = vcat(map(p->reshape(p,:),params(m))...)

anynan(x) = any(isnan.(x))

accuracy(x,y,model,n) = mean(onecold(cpu(model(x)),0:(n-1)) .== onecold(cpu(y),0:(n-1)))


function getData(args,indirs,ntype,train_dataset::Dataset,test_dataset::Dataset)
    nsets = floor(Int,args.n_train_evts/args.batch_size)
    train_set = []
    for i in 1:nsets
        train_x = zeros(UInt16, (args.nx,args.ny,args.n_samples*2,ntype*args.batch_size))
        train_y = zeros(UInt8, (ntype*args.batch_size))
        fillDataArrays(train_x,train_y,indirs,train_dataset,args.batch_size,args.nx,train_dataset,args.n_samples)
        push!(train_set,(train_x,onehotbatch(train_y,0:(ntype-1))))
    end
    test_x = zeros(UInt16, (args.nx,args.ny,args.n_samples*2,args.n_test_evts*ntype))
    test_y = zeros(UInt8, (args.n_test_evts*ntype))
    fillDataArrays(test_x,test_y,indirs,test_dataset,args.n_test_evts,args.nx,train_dataset,args.n_samples)
#    train_set = [makeMinibatch(train_x, train_y, i,ntype,args.nx,args.ny,args.n_samples) for i in mb_idxs]
#    test_set = makeMinibatch(test_x, test_y, 1:length(test_x),ntype,args.nx,args.ny,args.n_samples)
#    return train_set,test_set

    return train_set, (test_x,onehotbatch(test_y,0:(ntype-1)))

end

function buildBasicCNN(args, ntype)
    cnn_output_size = Int.(floor.([args.nx/8,args.ny/8,32]))
    return Chain(
    Conv((1, 1), floor(Int,args.n_samples*2)=>floor(Int,args.n_samples/8), pad=(0,0),stride=(1,1), relu),
    Conv((3, 3), floor(Int,args.n_samples/8)=>floor(Int,args.n_samples/8), pad=(1,1),stride=(1,1), relu),
    Conv((3, 3), floor(Int,args.n_samples/8)=>floor(Int,args.n_samples/16), pad=(0,0),stride=(2,2), relu),
    x -> reshape(x, :, size(x, 4)),
    Dense(270,135),
    Dense(135,32),
    Dense(32,ntype),
    softmax)
end


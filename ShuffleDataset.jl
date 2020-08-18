#=
ShuffleDataset:
- Julia version: 
- Author: blaine
- Date: 2020-08-18
=#
import JSON
using HDF5

function hasFiles(d::Dict)
    for key in keys(d)
        if size(d[key],1) > 0
            return true
    end
    return false
end

function fillQueue(q::Dict{int8,string}, fl::Dict{string,[]})

function main()
    if size(ARGS,1) < 2
        println("usage: julia ShuffleDataset.jl [<input directory1>, <input directory2>, ...]")
        exit(500)
    end
    mask = "WaveformPairSim.h5"
    n_events::int16 = 16384
    chunksize::int16 = 1024
    indirs = ARGS #input directories
    filesdict = Dict()
    dirmap = Dict()
    i::int8 = 0
    for d in indirs
        dirmap[d] = i
        i++
        if !isdir(d)
            error("Error: argument " + d + " is not a directory")
        end
        files = readdir(d,join=true,sort=false)
        filesdict[d] = []
        for f in files
            if f.endswith(mask)
                filesdict[d].append!(f)
            end
        end
    end

    i = 0
    fileQueue = Dict{int8,string}
    for
    while hasFiles(filesdict)
        if

        filedata = Dict("0" => [], "1" => [])

end

@time main()

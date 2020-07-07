struct Dataset
    name::String
    eventmap::Dict{String,Tuple{Int64,Int64}}
    Dataset(n) = new(n,Dict{String,Tuple{Int64,Int64}}())
end

function addFile(d::Dataset,f::String)
    if !haskey(d.eventmap,f)
        d.eventmap[f] = (0,0)
    end
end

function setEventRange(d::Dataset,f::String,l::Tuple{Int64,Int64})
    d.eventmap[f] = l
end

function print(d::Dataset)
    println(d.name)
    for key in d.eventmap
        println(String(key,": events ", d.eventmap[key][1], " - ",d.eventmap[key][2]))
    end
end

function writeToFile(d::Dataset,fname::String)
    open(fname, "w") do f
        i = 0
        for key in d.eventmap
            if i == 0 
                write(f, String(key,",", d.eventmap[key][1], ",",d.eventmap[key][2]))
            else
                write(f, String("\n",key,",", d.eventmap[key][1], ",",d.eventmap[key][2]))
            end
            i+=1
        end
    end
end

function hasFile(d::Dataset,fname::String)
    return haskey(d.eventmap,fname)
end

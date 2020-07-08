#=
TestPSD:
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

cd(@__DIR__)
@time test()

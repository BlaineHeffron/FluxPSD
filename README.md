# FluxPSD
particle discrimination in PROSPECT using Flux, a Julia ML library

To install prerequisites:

enter a julia prompt
type ] to go into the package window
run the following commands:
add HDF5
add Flux
add CUDA
add Parameters
add BSON
add SparseArrays


to run

julia TrainPSD.jl /path/to/dataset1 /path/to/dataset2 ...

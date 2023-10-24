import Pkg
Pkg.activate(".")
# Pkg.add("CUDA")
Pkg.update()
Pkg.status()
# Pkg.instantiate()
println("Environment activated, loading packages")
using Clapeyron
import Clapeyron: a_res, saturation_pressure, pressure

using Flux
using Plots, Statistics
using ForwardDiff, DiffResults

using Zygote, ChainRulesCore
using ImplicitDifferentiation

using CSV, DataFrames
using MLUtils
using RDKitMinimalLib 
using JLD2
using CUDA
println("Packages loaded, running SaftVRMieNN")
include("./saftvrmienn.jl")
println("SaftVRMieNN loaded, running script")
include("./10_gpu_training.jl")
println("Script finished executing")
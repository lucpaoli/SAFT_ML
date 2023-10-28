import Pkg
Pkg.activate(".")
# Pkg.update()
# Pkg.instantiate()
println("Environment activated, loading script")
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
println("Running 'saftvrmienn.jl'")
include("./saftvrmienn.jl")
# include("./7_cluster_training.jl")
println("Including training scripts")
include("./8_multithreaded_training.jl")
println("Running main")
main()
println("Script finished executing")
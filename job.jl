import Pkg
Pkg.activate(".")
# using Logging

println("Environment activated, loading script")
flush(stdout)

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
flush(stdout)
include("./saftvrmienn.jl")
# include("./7_cluster_training.jl")
println("Including training scripts")
flush(stdout)
# include("./8_multithreaded_training.jl")
# include("./12_ML_SAFT_architecture.jl")
# include("./13_ML_SAFT_loss2.jl")
# include("./15_NN_psat_vsat.jl")
include("./16_transfer_learning.jl")
println("Running main")
flush(stdout)
@time main()
println("Script finished executing")
flush(stdout)
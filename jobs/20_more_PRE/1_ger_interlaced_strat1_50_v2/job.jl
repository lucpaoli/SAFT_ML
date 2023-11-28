import Pkg
Pkg.activate("../../../.")
Pkg.instantiate()

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
# println("Pre-training network")
# flush(stdout)
# include("./pcpsaft_main.jl")
# @time main()

include("./main.jl")
println("Running main")
flush(stdout)
@time main()
println("Script finished executing")
flush(stdout)
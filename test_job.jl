import Pkg 
Pkg.activate(".")
println("heyyy~")
# using Revise
# using Base.Threads: @spawn, @sync, SpinLock
using Clapeyron
include("./saftvrmienn.jl")
# These are functions we're going to overload for SAFTVRMieNN
import Clapeyron: a_res, saturation_pressure, pressure

using Flux
using Plots, Statistics
using ForwardDiff, DiffResults

using Zygote#, ChainRulesCore
using ImplicitDifferentiation

using CSV, DataFrames
using MLUtils
using RDKitMinimalLib
using JLD2

# Multithreaded loss
using Zygote: bufferfrom
using Base.Threads: @spawn
println("Done!!!")

include("./8_multithreaded_training.jl")

println("Super Done, sleeping for 1 minute")
flush(stdout)
# flush(stderr)
sleep(60*1)
println("Done fr!")

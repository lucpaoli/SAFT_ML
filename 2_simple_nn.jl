import Pkg; Pkg.activate(".")
using Revise

# Equations of state
using Clapeyron
includet("./saftvrmienn.jl")
import Clapeyron: a_res

# Generating molecular feature vectors
using RDKitMinimalLib 

# Machine learning
using Flux
using ForwardDiff, DiffResults
using Zygote, ChainRulesCore

# Misc
using Statistics, Random, Plots

Random.seed!(1234)


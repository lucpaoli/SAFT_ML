using Clapeyron
include("../../../saftvrmienn.jl")
# These are functions we're going to overload for SAFTVRMieNN
import Clapeyron: a_res, saturation_pressure, pressure

using MolecularGraph, Graphs
using GraphNeuralNetworks
using MLUtils
using OneHotArrays
using Statistics, Random

using Flux
using Plots, Statistics
using ForwardDiff, DiffResults

using Zygote, ChainRulesCore
using ImplicitDifferentiation

using CSV, DataFrames
using MLUtils
using RDKitMinimalLib
using JLD2

# Multithreaded loss
using Zygote: bufferfrom
using Base.Threads: @spawn
using Plots

function make_graph_from_smiles(smiles::String)
    molgraph = smilestomol(smiles)

    g = SimpleGraph(nv(molgraph))
    for e in edges(molgraph)
        add_edge!(g, e.src, e.dst)
    end

    # Should number of hydrogens be one-hot encoded?
    f(vec, enc) = hcat(map(x -> onehot(x, enc), vec)...)
    num_h = f(implicit_hydrogens(molgraph), [0, 1, 2, 3, 4])
    hybrid = f(hybridization(molgraph), [:sp, :sp2, :sp3])
    atoms = f(atom_symbol(molgraph), [:C, :O, :N])

    # Node data should be matrix (num_features, num_nodes)
    # Matrix has num_nodes columns, num_features rows
    ndata = Float32.(vcat(num_h, hybrid, atoms))

    b_order = Float32.(f(bond_order(molgraph), [1, 2, 3]))
    edata = nothing #! Can use bond order later
    
    g = GNNGraph(g, ndata = ndata, edata = edata)
    return g
end

# Generate training set for liquid density and saturation pressure
function create_data(; batch_size=16, n_points=25)
    contains_only_c(name) = all(letter -> lowercase(letter) == 'c', name)

    # Create training & validation data
    # df = CSV.read("./pcpsaft_params/SI_pcp-saft_parameters.csv", DataFrame, header=1)
    df = CSV.read("../../../pcpsaft_params/training_data.csv", DataFrame, header=1)

    # Take only linear alkanes
    filter!(row -> occursin("Alkane", row.family), df)
    filter!(row -> contains_only_c(row.isomeric_SMILES), df)
    # sort!(df, :Mw)

    # df = first(df, 20) #* Take only first molecule in dataframe
    @show df.species
    mol_data = zip(df.species, df.isomeric_SMILES, df.Mw)
    println("Generating data for $(length(mol_data)) molecules...")

    T1 = GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}
    # g, T, Mw, name
    X_data = Tuple{T1, Float64, Float64, String}[]
    Y_data = Vector{Vector{Float64}}()

    # graph_dict = Dict{String, Tuple{T1, Float64}}()

    for (name, smiles, Mw) in mol_data
        saft_model = PPCSAFT([name])
        Tc, pc, Vc = crit_pure(saft_model)

        g = make_graph_from_smiles(smiles)
        # graph_dict[name] = (g, Mw)

        T_range = range(0.5 * Tc, 0.975 * Tc, n_points)
        for T in T_range
            (p_sat, Vₗ_sat, Vᵥ_sat) = saturation_pressure(saft_model, T)

            push!(X_data, (g, T, Mw, name)) 
            push!(Y_data, [Vₗ_sat, p_sat])
        end
    end

    Random.seed!(1234)
    #* Not sure what getobs does
    train_data, test_data = splitobs((X_data, Y_data), at = 1.0, shuffle = true) |> getobs

    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=false)

    println("n_batches = $(length(train_loader)), batch_size = $batch_size")

    return train_loader, test_loader
end

# todo split into two functions; parameter generation and Vₗ, p_sat calculation
function calculate_saft_parameters(model, g, Mw)
    λ_a = 6.0
    pred_params = model(g, g.ndata.x)

    # Add parameter bounding w tanh
    #   1 < m < 10
    #   2 < σ < 10
    #   5 < λ_r < 30
    # 100 < ϵ < 500
    l = [1.0, 2, 10, 100]
    u = [5.0, 6, 25, 500]
    c = [1.0, 1, 10, 100]
    biased_params = @. (u - l)/2.0 * (tanh(c * pred_params / u) + 1) + l

    saft_input = vcat(Mw, biased_params[1:2], [λ_a], biased_params[3:4])
    return saft_input
end

function SAFT_head(model, X)
    g, T, Mw, name = X

    # g, Mw = ignore_derivatives() do
    #     @atomic graph_dict[name]
    # end

    saft_input = calculate_saft_parameters(model, g, Mw)

    Tc = ignore_derivatives() do
        critical_temperature_NN(saft_input)
    end
    # todo include saturation volumes in loss
    if T < Tc
        sat_p = saturation_pressure_NN(saft_input, T)
        #! This repeats the volume root calculation
        Vₗ = volume_NN(saft_input, sat_p, T)
        ŷ_1 = !isnan(Vₗ) ? Vₗ : nothing
        ŷ_2 = !isnan(sat_p) ? sat_p : nothing

        ŷ = [ŷ_1, ŷ_2]
    else
        ŷ = [nothing, nothing]
    end

    return ŷ
end

function eval_loss(X_batch, y_batch, metric, model)
    batch_loss = 0.0
    n = 0
    for (X, y_vec) in zip(X_batch, y_batch)
        ŷ_vec = SAFT_head(model, X)

        for (ŷ, y) in zip(ŷ_vec, y_vec)
            if !isnothing(ŷ)
                batch_loss += metric(y, ŷ)
                n += 1
            end
        end

    end
    if n > 0
        batch_loss /= n
    end
    n_failed = length(y_batch) * 2 - n
    print(" $n_failed,")
    return batch_loss
end

function eval_loss_par(X_batch, y_batch, metric, model, n_chunks)
    print("n_failed =")
    n = length(X_batch)
    chunk_size = n ÷ n_chunks

    p = bufferfrom(zeros(n_chunks))

    # Creating views for each chunk
    X_chunks = vcat([view(X_batch, (i-1)*chunk_size+1:i*chunk_size) for i in 1:n_chunks-1], [view(X_batch, (n_chunks-1)*chunk_size+1:n)])
    y_chunks = vcat([view(y_batch, (i-1)*chunk_size+1:i*chunk_size) for i in 1:n_chunks-1], [view(y_batch, (n_chunks-1)*chunk_size+1:n)])

    @sync begin
        for i = 1:n_chunks
            @spawn begin
                p[i] = eval_loss(X_chunks[i], y_chunks[i], metric, model)
            end
        end
    end
    print("\n")
    flush(stdout)
    return sum(p) / n_chunks # average partial losses
end

function percent_error(y, ŷ)
    return 100 * abs(y - ŷ) / y
end

function mse(y, ŷ)
    return ((y - ŷ) / y)^2
end

function train_model!(model, train_loader, test_loader, optim; epochs=10, log_filename="params_log_nodict.csv")
    for epoch in 1:epochs
        epoch_start_time = time()

        batch_loss = 0.0
        unique_fps = Dict()

        for (X_batch, y_batch) in train_loader
            for (g, T, Mw, name) in X_batch
                if !haskey(unique_fps, name)
                    unique_fps[name] = (g, Mw)
                end
            end

            loss, grads = Flux.withgradient(model) do m
                loss = eval_loss_par(X_batch, y_batch, mse, m, Threads.nthreads())
                loss
            end
            batch_loss += loss
            @assert !isnan(loss)
            @assert !iszero(loss)

            Flux.update!(optim, model, grads[1])
        end

        for (name, (g, Mw)) in unique_fps
            Mw, m, σ, λ_a, λ_r, ϵ = calculate_saft_parameters(model, g, Mw)

            # Log to file as csv, formatted as:
            # epoch, molecule, m, σ, λ_a, λ_r, ϵ
            open(log_filename, "a") do io
                write(io, "$epoch;$name;$Mw;$m;$σ;$λ_a;$λ_r;$ϵ\n")
            end
        end

        batch_loss /= length(train_loader)
        epoch_duration = time() - epoch_start_time

        epoch % 1 == 0 && println("\nepoch $epoch: batch_loss = $batch_loss, time = $(epoch_duration)s")
        flush(stdout)
    end
end

# model(g) = m, σ, λ_r, ϵ
function create_graphattention_model(nin, nh; nout=4, ngclayers=3, nhlayers=3, afunc=relu)
    GNNChain(
        GATv2Conv(nin => nh, afunc),
        [GATv2Conv(nh => nh, afunc) for _ in 1:ngclayers]...,
        GlobalPool(mean),
        # Dropout(0.2),
        [Dense(nh => nh, afunc) for _ in 1:nhlayers]...,
        Dense(nh, nout, x -> x),
    )
end

function main(; epochs=1000)
    train_loader, test_loader = create_data(n_points=50, batch_size=230)

    # How to determine nin? I think it's 11
    nin = 11
    nh = 512

    # model = create_graphconv_model(nin, nh)
    model = create_graphattention_model(nin, nh)

    println("training on $(Threads.nthreads()) threads")
    flush(stdout)

    optim = Flux.setup(Flux.Adam(1e-3), model)
    train_model!(model, train_loader, test_loader, optim; epochs=epochs)
end

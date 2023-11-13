using Clapeyron
include("../../../saftvrmienn.jl")
# These are functions we're going to overload for SAFTVRMieNN
import Clapeyron: a_res, saturation_pressure, pressure

using MolecularGraph, Graphs
# using MLUtils
using GeometricFlux
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
    
    # g = GNNGraph(g, ndata = ndata, edata = edata)
    return g
end

# Generate training set for liquid density and saturation pressure
function create_data(; batch_size=16, n_points=25, pretraining=false)
    contains_only_c(name) = all(letter -> lowercase(letter) == 'c', name)

    # Create training & validation data
    df = CSV.read("../../../pcpsaft_params/training_data.csv", DataFrame, header=1)

    #* Only alkanes
    filter!(row -> occursin("Alkane", row.family), df)

    @show df.species
    mol_data = zip(df.species, df.isomeric_SMILES, df.Mw)
    println("Generating data for $(length(mol_data)) molecules...")

    if pretraining
        T1 = Float64
        T2 = GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}
        X_data = Vector{Tuple{T2,T1}}()
        Y_data = Vector{Vector{T1}}()

        for (name, smiles, Mw) in mol_data
            g = make_graph_from_smiles(smiles)

            saft_model = PPCSAFT([name])
            m = saft_model.params.segment.values[1]
            sigma = saft_model.params.sigma.values[1] * 1e10
            λ_r = 15.0
            epsilon = saft_model.params.epsilon.values[1]

            push!(X_data, (g, Mw))
            push!(Y_data, [m, sigma, λ_r, epsilon])
        end
    else
        T1 = Float64
        T2 = GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}
        X_data = Vector{Tuple{T2,T1,T1,String}}([])
        Y_data = Vector{Vector{T1}}()

        for (name, smiles, Mw) in mol_data
            saft_model = PPCSAFT([name])
            Tc, pc, Vc = crit_pure(saft_model)

            g = make_graph_from_smiles(smiles)

            T_range = range(0.5 * Tc, 0.975 * Tc, n_points)
            for T in T_range
                (p_sat, Vₗ_sat, Vᵥ_sat) = saturation_pressure(saft_model, T)

                push!(X_data, (g, T, Mw, name))
                push!(Y_data, [Vₗ_sat, p_sat])
            end
        end
    end

    train_data, test_data = splitobs((X_data, Y_data), at=1.0, shuffle=false)

    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=false)
    println("n_batches = $(length(train_loader)), batch_size = $batch_size")
    return train_loader, test_loader
end

function calculate_saft_parameters(model, g, Mw)
    λ_a = 6.0
    pred_params = model(g, g.ndata.x)

    m, σ, λ_r, ϵ = pred_params

    f(x, c) = elu(x-c, 1e-3) + c

    m = f(m, 1.0)
    σ = f(σ, 2.0)
    λ_r = 10 * f(λ_r, 1.0)
    ϵ = 100 * f(ϵ, 1.0)

    saft_input = [Mw, m, σ, λ_a, λ_r, ϵ]

    return saft_input
end

function SAFT_head(model, X)
    fp, T, Mw, name = X

    saft_input = calculate_saft_parameters(model, fp, Mw)

    sat_p = saturation_pressure_NN(saft_input, T)
    #! This repeats the volume root calculation
    Vₗ = volume_NN(saft_input, sat_p, T)

    ŷ_1 = !isnan(Vₗ) ? Vₗ : nothing
    ŷ_2 = !isnan(sat_p) ? sat_p : nothing

    return [ŷ_1, ŷ_2]
end

function eval_loss(X_batch, y_batch, metric, model, use_saft_head)
    batch_loss = 0.0
    n = 0
    
    for (X, y_vec) in zip(X_batch, y_batch)
        if use_saft_head
            ŷ_vec = SAFT_head(model, X)
        else
            # ŷ_vec = model(X[1])
            # (fp, Mw), [m, sigma, λ_r, epsilon]
            fp, Mw = X
            saft_params = calculate_saft_parameters(model, X...)
            ŷ_vec = [saft_params[i] for i in [2, 3, 5, 6]]
        end

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

function eval_loss_par(X_batch, y_batch, metric, model, n_chunks, use_saft_head)
    print("n_failed =")
    # flush(stdout)
    n = length(X_batch)
    chunk_size = n ÷ n_chunks

    p = bufferfrom(zeros(n_chunks))

    # Creating views for each chunk
    X_chunks = vcat([view(X_batch, (i-1)*chunk_size+1:i*chunk_size) for i in 1:n_chunks-1], [view(X_batch, (n_chunks-1)*chunk_size+1:n)])
    y_chunks = vcat([view(y_batch, (i-1)*chunk_size+1:i*chunk_size) for i in 1:n_chunks-1], [view(y_batch, (n_chunks-1)*chunk_size+1:n)])

    @sync begin
        for i = 1:n_chunks
            @spawn begin
                p[i] = eval_loss(X_chunks[i], y_chunks[i], metric, model, use_saft_head)
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

function train_model!(model, train_loader, test_loader, optim; epochs=10, pretraining=false)
    nthreads = pretraining ? min(10, Threads.nthreads()) : Threads.nthreads()
    log_filename = pretraining ? "params_log_pretraining.csv" : "params_log.csv" 

    println("training on $nthreads threads")
    flush(stdout)

    for epoch in 1:epochs
        epoch_start_time = time() # Start timing the epoch
        batch_loss = 0.0
        unique_fps = Dict()

        for (X_batch, y_batch) in train_loader

            if !pretraining
                # todo print how many datapoints masked out per iter
                Tc_dict = Dict{String, Float64}()
                for (fp, T, Mw, name) in X_batch
                    if !haskey(unique_fps, name)
                        unique_fps[name] = (fp, Mw)
                    end
                    if !haskey(Tc_dict, name)
                        saft_input = calculate_saft_parameters(model, fp, Mw)
                        Tc = critical_temperature_NN(saft_input)
                        Tc_dict[name] = Tc
                    end
                end
                indices = findall(entry -> entry[2] < Tc_dict[entry[4]], X_batch)
                X_batch, y_batch = X_batch[indices], y_batch[indices]
            end

            loss, grads = Flux.withgradient(model) do m
                loss = eval_loss_par(X_batch, y_batch, mse, m, nthreads, !pretraining)
                loss
            end
            batch_loss += loss
            @assert !isnan(loss)

            Flux.update!(optim, model, grads[1])
        end

        # Log params to file
        for (name, (fp, Mw)) in unique_fps
            Mw, m, σ, λ_a, λ_r, ϵ = calculate_saft_parameters(model, fp, Mw)

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

function create_model(nin, nh; nout=4, ngclayers=2, nhlayers=2)
    GNNChain(
        CGConv(nin => nh, relu),
        [CGConv(nh => nh, relu; residual=true) for _ in 1:ngclayers]...,
        GlobalPool(mean),
        [Dense(nh, nh, relu) for _ in 1:nhlayers]...,
        Dense(nh, nout, x -> x),
    )
end

function main(; epochs=5000)
    Random.seed!(622)
    model = main_pcpsaft()

    train_loader, test_loader = create_data(n_points=50, batch_size=400)
    @show n_features = length(first(train_loader)[1][1][1])

    optim = Flux.setup(Flux.Adam(1e-5), model)
    train_model!(model, train_loader, test_loader, optim; epochs=epochs)
end

function main_pcpsaft(; epochs=1000)
    train_loader, test_loader = create_data(n_points=50, batch_size=8, pretraining=true)
    @show n_features = length(first(train_loader)[1][1][1])
    nin = 11
    nh = 30

    model = create_model(nin, nh)
    optim = Flux.setup(Flux.Adam(1e-3), model)
    train_model!(model, train_loader, test_loader, optim; epochs=epochs, pretraining=true)

    return model
end
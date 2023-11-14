using Clapeyron
include("../../../saftvrmienn.jl")
# These are functions we're going to overload for SAFTVRMieNN
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

# Multithreaded loss
using Zygote: bufferfrom
using Base.Threads: @spawn
using Plots
using Random

function make_fingerprint(s::String)::Vector{Float64}
    mol = get_mol(s)
    @assert !isnothing(mol)

    fp = []

    fp_str_morgan = get_morgan_fp(mol, Dict{String,Any}("radius"=> 5, "nbits" => 1024))
    fp_str_atom_pair = get_atom_pair_fp(mol, Dict{String,Any}("radius"=> 6, "nbits" => 1024))
    fp_str_pattern = get_pattern_fp(mol, Dict{String,Any}("radius"=> 7, "nbits" => 1024))

    fp_str = fp_str_morgan * fp_str_atom_pair * fp_str_pattern

    append!(fp, [parse(Float64, string(c)) for c in fp_str])

    desc = get_descriptors(mol)
    relevant_keys = [
        "CrippenClogP",
        "NumHeavyAtoms",
        "amw",
        "FractionCSP3",
    ]
    relevant_desc = [desc[k] for k in relevant_keys]
    append!(fp, relevant_desc)

    return fp
end

# Generate training set for liquid density and saturation pressure
function create_data(; batch_size=16, n_points=25, pretraining=false)
    contains_only_c(name) = all(letter -> lowercase(letter) == 'c', name)

    # Create training & validation data
    df = CSV.read("../../../pcpsaft_params/training_data.csv", DataFrame, header=1)

    #* Only alkanes
    filter!(row -> occursin("Alkane", row.family), df)
    #* Only linear alkanes
    # filter!(row -> contains_only_c(row.isomeric_SMILES), df)

    @show df.species
    mol_data = zip(df.species, df.isomeric_SMILES, df.Mw)
    println("Generating data for $(length(mol_data)) molecules...")

    if pretraining
        T = Float64
        X_data = Vector{Tuple{Vector{T},Nothing,T,String}}()
        Y_data = Vector{Vector{T}}()

        for (name, smiles, Mw) in mol_data
            fp = make_fingerprint(smiles)
            append!(fp, Mw)

            saft_model = PPCSAFT([name])
            m = saft_model.params.segment.values[1]
            sigma = saft_model.params.sigma.values[1] * 1e10
            λ_r = 15.0
            epsilon = saft_model.params.epsilon.values[1]

            println("name = $name, y_data = $([m, sigma, λ_r, epsilon])")

            push!(X_data, (fp, nothing, Mw, name))
            push!(Y_data, [m, sigma, λ_r, epsilon])
        end
    else
        T = Float64
        X_data = Vector{Tuple{Vector{T},T,T,String}}([])
        Y_data = Vector{Vector{T}}()

        for (name, smiles, Mw) in mol_data
            saft_model = PPCSAFT([name])
            Tc, pc, Vc = crit_pure(saft_model)

            fp = make_fingerprint(smiles)
            append!(fp, Mw)

            T_range = range(0.5 * Tc, 0.975 * Tc, n_points)
            for T in T_range
                (p_sat, Vₗ_sat, Vᵥ_sat) = saturation_pressure(saft_model, T)

                push!(X_data, (fp, T, Mw, name))
                push!(Y_data, [Vₗ_sat, p_sat])
            end
        end
    end

    #* Remove columns from fingerprints
    # Identify zero & one columns
    for num = [0, 1]
        num_cols = length(X_data[1][1])
        zero_cols = trues(num_cols)
        for (vec, _...) in X_data
            zero_cols .&= (vec .== num)
        end
        keep_cols = .!zero_cols # Create a Mask
        X_data = [(vec[keep_cols], vals...) for (vec, vals...) in X_data] # Apply Mask
    end

    train_data, test_data = splitobs((X_data, Y_data), at=1.0, shuffle=false)

    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=false)
    println("n_batches = $(length(train_loader)), batch_size = $batch_size")
    return train_loader, test_loader
end

function calculate_saft_parameters(model, fp, Mw)
    λ_a = 6.0
    pred_params = model(fp)

    # f(x, u, l, c) = (u - l)/2.0 * (tanh(c * x / u) + 1) + l
    # σ = f(σ, 10, 2, 1)
    # λ_r = f(λ_r, 100, 10, 10)
    # ϵ = f(ϵ, 0, 500, 100)

    # b = [2.5, 3.5, 12.0, 250.0]
    # c = [1.0, 1, 10, 100]
    # biased_params = @. pred_params * c + b

    m, σ, λ_r, ϵ = pred_params

    # f(x, c) = elu(x-c, 1e-3) + c
    f(x, c) = elu(x, 0.05) + c

    m = f(m, 1.0)
    σ = f(σ, 2.0)
    λ_r = 10.0 * f(λ_r, 1.0)
    ϵ = 100.0 * f(ϵ, 1.0)

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
            fp, T, Mw, name = X
            ŷ = calculate_saft_parameters(model, fp, Mw)
            ŷ_vec = [ŷ[2], ŷ[3], ŷ[5] + 4*randn(), ŷ[6]]
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
    open(log_filename, "a") do io
        write(io, "epoch;name;Mw;m;σ;λ_a;λ_r;ϵ\n")
    end

    println("training on $nthreads threads")
    flush(stdout)

    for epoch in 1:epochs
        epoch_start_time = time() # Start timing the epoch
        batch_loss = 0.0
        unique_fps = Dict()

        for (X_batch, y_batch) in train_loader
            Tc_dict = Dict{String, Float64}()
            for (fp, T, Mw, name) in X_batch
                if !haskey(unique_fps, name)
                    unique_fps[name] = (fp, Mw)
                end
                if !haskey(Tc_dict, name) && !pretraining
                    saft_input = calculate_saft_parameters(model, fp, Mw)
                    Tc = critical_temperature_NN(saft_input)
                    Tc_dict[name] = Tc
                end
            end
            if !pretraining
                original_size = length(X_batch)
                indices = findall(entry -> entry[2] < Tc_dict[entry[4]], X_batch)
                X_batch, y_batch = X_batch[indices], y_batch[indices]
                n_masked = original_size - length(X_batch)
                println("n_masked = $n_masked")
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

function create_ff_model(nfeatures)
    nout = 4
    return Chain(
        Dense(nfeatures, nout*8, leakyrelu),
        Dense(nout*8, nout*4, leakyrelu),
        Dense(nout*4, nout*2, leakyrelu),
        Dense(nout*2, nout, x -> x),
    )
end

function main(; epochs=5000)
    Random.seed!(1234)
    model = main_pcpsaft()

    train_loader, test_loader = create_data(n_points=50, batch_size=400)
    @show n_features = length(first(train_loader)[1][1][1])

    optim = Flux.setup(Flux.Adam(1e-2), model)
    train_model!(model, train_loader, test_loader, optim; epochs=epochs)
end

function main_pcpsaft(; epochs=50)
    train_loader, test_loader = create_data(n_points=50, batch_size=8, pretraining=true)
    @show n_features = length(first(train_loader)[1][1][1])

    model = create_ff_model(n_features)
    optim = Flux.setup(Flux.Adam(1e-3), model)
    train_model!(model, train_loader, test_loader, optim; epochs=epochs, pretraining=true)

    return model
end
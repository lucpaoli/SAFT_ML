import Pkg;
Pkg.activate(".");
using Clapeyron
include("./saftvrmienn.jl")
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

# Generate training set for liquid density and saturation pressure
function create_data(; batch_size=16, n_points=25)
    # Create training & validation data
    df = CSV.read("./pcpsaft_params/SI_pcp-saft_parameters.csv", DataFrame, header=1)
    filter!(row -> occursin("Alkane", row.family), df)
    df = first(df, 2) #* Take only first molecule in dataframe
    @show df.common_name
    mol_data = zip(df.common_name, df.isomeric_smiles, df.molarweight)
    println("Generating data for $(length(mol_data)) molecules...")

    function make_fingerprint(s::String)::Vector{Float64}
        mol = get_mol(s)
        @assert !isnothing(mol)

        fp = []
        # for (nbits, rad) in [(256, 256), (1, 3)]
        #* Approximately ECFP4 fingerprint
        nbits = 256
        rad = 4

        fp_details = Dict{String,Any}("nBits" => nbits, "radius" => rad)
        fp_str = get_morgan_fp(mol, fp_details)
        append!(fp, [parse(Float64, string(c)) for c in fp_str])
        # end

        desc = get_descriptors(mol)
        relevant_keys = [
            "CrippenClogP",
            "NumHeavyAtoms",
            "amw",
            "FractionCSP3",
        ]
        relevant_desc = [desc[k] for k in relevant_keys]
        append!(fp, last.(relevant_desc))

        return fp
    end

    T = Float64
    # X_data = Vector{Tuple{Vector{T},T,T,T}}([])
    X_data = Vector{Tuple{Vector{T},T,T}}([])
    Y_data = Vector{Vector{T}}()

    # n = 0
    for (name, smiles, Mw) in mol_data
        saft_model = PPCSAFT([name])
        # saft_model = SAFTVRMie([name])
        Tc, pc, Vc = crit_pure(saft_model)

        fp = make_fingerprint(smiles)
        # fp = [1.0]
        append!(fp, Mw)

        T_range = range(0.5 * Tc, 0.975 * Tc, n_points)
        for T in T_range
            (p_sat, Vₗ_sat, Vᵥ_sat) = saturation_pressure(saft_model, T)

            # p = p_sat * 5.0
            # Vₗ = volume(saft_model, p, T; phase=:liquid)

            push!(X_data, (fp, T, Mw))
            push!(Y_data, [Vₗ_sat, p_sat])
        end
    end

    #* Remove columns from fingerprints
    # Identify zero & one columns
    for num = [0, 1]
        num_cols = length(X_data[1][1])
        zero_cols = trues(num_cols)
        for (vec, _, _) in X_data
            zero_cols .&= (vec .== num)
        end
        keep_cols = .!zero_cols # Create a Mask
        X_data = [(vec[keep_cols], vals...) for (vec, vals...) in X_data] # Apply Mask
    end

    train_data, test_data = splitobs((X_data, Y_data), at=1.0, shuffle=false)

    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=false)
    test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=false)
    println("n_batches = $(length(train_loader)), batch_size = $batch_size")
    flush(stdout)
    return train_loader, test_loader
end


function create_ff_model(nfeatures)
    # Base NN architecture from "Fitting Error vs Parameter Performance"
    nout = 4
    # model = Chain(
    #     Dense(nfeatures, nout, x -> x; bias=false, init=zeros32),
    # )
    #* glorot_uniform default initialisation
    #! zeros32 is probably _really bad_ as an initialisation, but glorot_uniform can lead to invalid SAFT inputs
    model = Chain(
        Dense(nfeatures, nout * 8, tanh, init=Flux.glorot_normal),
        Dense(nout * 8, nout * 4, tanh, init=Flux.glorot_normal),
        Dense(nout * 4, nout * 2, tanh, init=Flux.glorot_normal),
        Dense(nout * 2, nout, x -> x, init=Flux.glorot_normal), # Allow unbounded negative outputs; parameter values physically bounded in SAFT layer
    )
    # model(x) = m, σ, λ_a, λ_r, ϵ

    # return nn_model, unbounded_model
    return model
end

function get_idx_from_iterator(iterator, idx)
    data_iterator = iterate(iterator)
    for _ in 1:idx-1
        data_iterator = iterate(iterator, data_iterator[2])
    end
    return data_iterator[1]
end


function SAFT_head(model, X; b=[2.5, 3.5, 12.0, 250.0], c=Float64[1, 1, 10, 100])
    fp, T, Mw = X

    # m = 1.8514
    # σ = 4.0887
    λ_a = 6.0
    # λ_r = 13.65
    # ϵ = 273.64
    # fp, p, T, Mw = X
    pred_params = model(fp)

    # Add bias and scale
    biased_params = @. pred_params * c + b

    m, σ, λ_r, ϵ = biased_params
    m = max(1.0, m)

    # saft_input = vcat(Mw, biased_params[1:2], [λ_a], biased_params[3:4])
    saft_input = [Mw, m, σ, λ_a, λ_r, ϵ]

    Tc = ignore_derivatives() do
        critical_temperature_NN(saft_input)
    end
    # todo include saturation volumes in loss
    if T < Tc
        sat_p = saturation_pressure_NN(saft_input, T)
        #! This repeats the volume root calculation
        Vₗ = volume_NN(saft_input, sat_p, T)

        ŷ = [Vₗ, sat_p]
    else
        #* Instead of ignoring, could:
        #* - compare to critical point
        #* - compare to sat_p, Vₗ at reduced T (T/Tc^exp vs T/Tc)
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
    # penalize batch_loss depending on how many failed
    # n_failed = length(y_batch) - n
    # batch_loss += n_failed * 1000.0

    return batch_loss
end

function eval_loss_par(X_batch, y_batch, metric, model, n_chunks)
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
    return sum(p) / n_chunks # average partial losses
end

function percent_error(y, ŷ)
    return 100 * abs(y - ŷ) / y
end

function mse(y, ŷ)
    return ((y - ŷ) / y)^2
end

function train_model!(model, train_loader, test_loader; epochs=10)
    optim = Flux.setup(Flux.Adam(0.01), model) # 1e-3 usually safe starting LR
    # optim = Flux.setup(Descent(0.001), model)

    println("training on $(Threads.nthreads()) threads")
    flush(stdout)

    for epoch in 1:epochs
        batch_loss = 0.0
        for (X_batch, y_batch) in train_loader

            loss, grads = Flux.withgradient(model) do m
                loss = eval_loss_par(X_batch, y_batch, percent_error, m, Threads.nthreads())
                # loss = eval_loss(X_batch, y_batch, mse, m)
                loss
            end
            batch_loss += loss
            @assert !isnan(loss)

            Flux.update!(optim, model, grads[1])
        end
        batch_loss /= length(train_loader)
        epoch % 1 == 0 && println("epoch $epoch: batch_loss = $batch_loss")
        flush(stdout)
    end
end

function main(; epochs=1)
    train_loader, test_loader = create_data(n_points=50, batch_size=100) # Should make 5 batches / epoch. 256 / 8 gives 32 evaluations per thread
    @show n_features = length(first(train_loader)[1][1][1])

    model = create_ff_model(n_features)
    # @show model.layers[1].weight, model([1.0])
    train_model!(model, train_loader, test_loader; epochs=epochs)
    # save_model(model, "model_state.jld2")
    model_state = Flux.state(model)
    jldsave("model_state_all_alkanes.jld2"; model_state)

    return model
end


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

function create_data(; batch_size=16, n_points=25)
    # Create training & validation data
    df = CSV.read("./pcpsaft_params/SI_pcp-saft_parameters.csv", DataFrame, header=1)
    filter!(row -> occursin("Alkane", row.family), df)
    # df = first(df, 1) #* Take only first molecule in dataframe
    mol_data = zip(df.common_name, df.isomeric_smiles, df.molarweight)
    println("Generating data for $(length(mol_data)) molecules...")

    function make_fingerprint(s::String)::Vector{Float64}
        mol = get_mol(s)
        @assert !isnothing(mol)

        fp = []
        # for (nbits, rad) in [(256, 256), (1, 3)]
        #* Approximately ECFP4 fingerprint
        nbits = 4096
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
    X_data = Vector{Tuple{Vector{T},T,T}}([])
    Y_data = Vector{Vector{T}}()

    for (name, smiles, Mw) in mol_data
        try
            saft_model = PPCSAFT([name])
            Tc, pc, Vc = crit_pure(saft_model)

            fp = make_fingerprint(smiles)
            append!(fp, Mw)

            T_range = range(0.5 * Tc, 0.975 * Tc, n_points)
            for T in T_range
                (p₀, V_vec...) = saturation_pressure(saft_model, T)
                push!(X_data, (fp, T, Mw))
                push!(Y_data, [p₀])
            end
        catch
            println("Fingerprint generation failed for $name")
        end
    end

    #* Remove columns from fingerprints
    # Identify zero & one columns
    num_cols = length(X_data[1][1])
    zero_cols = trues(num_cols)
    for (vec, _, _) in X_data
        zero_cols .&= (vec .== 0)
    end
    keep_cols = .!zero_cols # Create a Mask
    X_data = [(vec[keep_cols], val1, val2) for (vec, val1, val2) in X_data] # Apply Mask

    num_cols = length(X_data[1][1])
    one_cols = trues(num_cols)
    for (vec, _, _) in X_data
        one_cols .&= (vec .== 0)
    end
    keep_cols = .!one_cols # Create a Mask
    X_data = [(vec[keep_cols], val1, val2) for (vec, val1, val2) in X_data] # Apply Mask

    train_data, test_data = splitobs((X_data, Y_data), at=0.8, shuffle=false)

    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=false)
    test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=false)
    println("n_batches = $(length(train_loader)), batch_size = $batch_size")
    flush(stdout)
    return train_loader, test_loader
end


function create_ff_model(nfeatures)
    # Base NN architecture from "Fitting Error vs Parameter Performance"
    nout = 5
    model = Chain(
        Dense(nfeatures, 1024, relu),
        Dense(1024, 512, relu),
        Dense(512, 256, relu),
        Dense(256, 128, relu),
        Dense(128, 64, relu),
        Dense(64, nout, relu),
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


function SAFT_head(model, X; b=[3.0, 3.5, 7.0, 12.5, 250.0], c=10.0)
    fp, T, Mw = X
    pred_params = model(fp)

    # Add bias and scale
    biased_params = pred_params / c + b

    saft_input = vcat(Mw, biased_params)
    Tc = ignore_derivatives() do
        critical_temperature_NN(saft_input)
    end
    if T < Tc
        sat_p = saturation_pressure_NN(saft_input, T)
        if !isnan(sat_p)
            ŷ = sat_p
        else
            # println("sat_p is NaN at T = $T, saft_input = $saft_input")
            ŷ = nothing
        end
    else
        ŷ = nothing
    end

    return ŷ
end

function eval_loss(X_batch, y_batch, metric, model)
    batch_loss = 0.0
    n = 0
    for (X, y_vec) in zip(X_batch, y_batch)
        y = y_vec[1]
        ŷ = SAFT_head(model, X)
        if !isnothing(ŷ)
            batch_loss += metric(y, ŷ)
            n += 1
        end
    end
    if n > 0 
        batch_loss /= n
    end
    # penalize batch_loss depending on how many failed
    # batch_loss += length(y_batch) - n

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
    return (y - ŷ)^2
end

function train_model!(model, train_loader, test_loader; epochs=10)
    optim = Flux.setup(Flux.Adam(0.001), model) # 1e-3 usually safe starting LR

    println("training on $(Threads.nthreads()) threads")
    flush(stdout)

    for epoch in 1:epochs
        batch_loss = 0.0
        for (X_batch, y_batch) in train_loader

            loss, grads = Flux.withgradient(model) do m
                loss = eval_loss_par(X_batch, y_batch, percent_error, m, Threads.nthreads())
                loss
            end
            batch_loss += loss

            Flux.update!(optim, model, grads[1])
        end
        batch_loss /= length(train_loader)
        epoch % 1 == 0 && println("epoch $epoch: batch_loss = $batch_loss")
        flush(stdout)
    end
end

function main()
    train_loader, test_loader = create_data(n_points=32, batch_size=512) # Should make 5 batches / epoch. 256 / 8 gives 32 evaluations per thread
    n_features = length(first(train_loader)[1][1][1])
    println("n_features = $n_features")
    flush(stdout)

    model = create_ff_model(n_features)
    train_model!(model, train_loader, test_loader; epochs=50)
end
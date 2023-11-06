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
function create_data(; batch_size=16)
    contains_only_c(name) = all(letter -> lowercase(letter) == 'c', name)

    # Create training & validation data
    # df = CSV.read("./pcpsaft_params/SI_pcp-saft_parameters.csv", DataFrame, header=1)
    df = CSV.read("./pcpsaft_params/training_data.csv", DataFrame, header=1)

    # Take only linear alkanes
    filter!(row -> occursin("Alkane", row.family), df)
    filter!(row -> contains_only_c(row.isomeric_SMILES), df) 
    # sort!(df, :Mw)

    # df = first(df, 20) #* Take only first molecule in dataframe
    @show df.species
    mol_data = zip(df.species, df.isomeric_SMILES, df.Mw)
    println("Generating data for $(length(mol_data)) molecules...")

    function make_fingerprint(s::String)::Vector{Float64}
        mol = get_mol(s)
        @assert !isnothing(mol)

        fp = []
        nbits = 256
        rad = 4

        fp_details = Dict{String,Any}("nBits" => nbits, "radius" => rad)
        fp_str = get_morgan_fp(mol, fp_details)
        append!(fp, [parse(Float64, string(c)) for c in fp_str])

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
    X_data = Vector{Tuple{Vector{T},String}}([])
    Y_data = Vector{Vector{T}}()

    for (name, smiles, Mw) in mol_data
        saft_model = PPCSAFT([name])
        Tc, pc, Vc = crit_pure(saft_model)

        fp = make_fingerprint(smiles)
        append!(fp, Mw)

        m = saft_model.params.segment.values[1]
        sigma = saft_model.params.sigma.values[1]
        epsilon = saft_model.params.epsilon.values[1]
        push!(X_data, (fp, name))
        push!(Y_data, [m, sigma, epsilon])
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

    # Only train data at the moment
    train_data, test_data = splitobs((X_data, Y_data), at=1.0, shuffle=false)

    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=false)
    test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=false)
    println("n_batches = $(length(train_loader)), batch_size = $batch_size")
    return train_loader, test_loader
end


function create_ff_model(nfeatures)
    # Base NN architecture from "Fitting Error vs Parameter Performance"
    nout = 4
    nout_ppcsaft = 3
    #* glorot_uniform default initialisation
    #* zeros32 is probably _really bad_ as an initialisation, but glorot_uniform can lead to invalid SAFT inputs
    model = Chain(
        Dense(nfeatures, nout * 8, tanh, init=Flux.glorot_normal),
        Dense(nout * 8, nout * 4, tanh, init=Flux.glorot_normal),
        Dense(nout * 4, nout * 2, tanh, init=Flux.glorot_normal),
        Dense(nout * 2, nout_ppcsaft, x -> x, init=Flux.zeros32), # Allow unbounded negative outputs; parameter values physically bounded in SAFT layer
    )
    # model(x) = m, σ, ϵ
    return model
end

function get_idx_from_iterator(iterator, idx)
    data_iterator = iterate(iterator)
    for _ in 1:idx-1
        data_iterator = iterate(iterator, data_iterator[2])
    end
    return data_iterator[1]
end

function eval_loss(X_batch, y_batch, metric, model)
    batch_loss = 0.0
    n = 0
    for (X, y_vec) in zip(X_batch, y_batch)
        fp, name = X
        ŷ_vec = model(fp)
        batch_loss += sum(metric(y_vec, ŷ_vec))
    end
    return batch_loss / n
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
    return @. 100 * abs(y - ŷ) / y
end

function mse(y, ŷ)
    return @. ((y - ŷ) / y)^2
end

function train_model!(model, train_loader, test_loader; epochs=10, log_filename="params_log_linear_alkanes_10k.csv")
    optim = Flux.setup(Flux.Adam(1e-3), model) # 1e-3 usually safe starting LR

    println("training on $(Threads.nthreads()) threads")
    flush(stdout)

    # todo report time for each epoch
    for epoch in 1:epochs
        epoch_start_time = time() # Start timing the epoch

        batch_loss = 0.0

        for (X_batch, y_batch) in train_loader

            loss, grads = Flux.withgradient(model) do m
                loss = eval_loss_par(X_batch, y_batch, mse, m, Threads.nthreads())
                # loss = eval_loss(X_batch, y_batch, mse, m)
                loss
            end
            batch_loss += loss
            @assert !isnan(loss)

            Flux.update!(optim, model, grads[1])
        end

        batch_loss /= length(train_loader)
        epoch_duration = time() - epoch_start_time

        epoch % 1 == 0 && println("\nepoch $epoch: batch_loss = $batch_loss, time = $(epoch_duration)s")
        flush(stdout)
    end
end

function main(; epochs=1000)
    #! 23 datapoints total
    train_loader, test_loader = create_data(batch_size=10) # Should make 5 batches / epoch. 256 / 8 gives 32 evaluations per thread
    @show n_features = length(first(train_loader)[1][1][1])

    model = create_ff_model(n_features)

    #! Train for 1k epochs directly on PPCSAFT data
    train_model!(model, train_loader, test_loader; epochs=epochs)
    model_state = Flux.state(model)
    jldsave("model_state_ppcsaft.jld2"; model_state)

    return model
end

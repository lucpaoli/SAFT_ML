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

# Generate training set for liquid density and saturation pressure
function create_data(; batch_size=16, n_points=25)
    contains_only_c(name) = all(letter -> lowercase(letter) == 'c', name)

    # Create training & validation data
    # df = CSV.read("./pcpsaft_params/SI_pcp-saft_parameters.csv", DataFrame, header=1)
    df = CSV.read("../../../pcpsaft_params/training_data.csv", DataFrame, header=1)

    # Take only linear alkanes
    filter!(row -> occursin("Alkane", row.family), df)
    # filter!(row -> contains_only_c(row.isomeric_SMILES), df)
    # sort!(df, :Mw)

    # df = first(df, 20) #* Take only first molecule in dataframe
    @show df.species
    mol_data = zip(df.species, df.isomeric_SMILES, df.Mw)
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
        append!(fp, relevant_desc)

        return fp
    end

    T = Float64
    # X_data = Vector{Tuple{Vector{T},T,T,T}}([])
    X_data = Vector{Tuple{Vector{T},T,T,T,String}}([])
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

            p = p_sat * 5.0
            # Vₗ = volume(saft_model, p, T; phase=:liquid)

            push!(X_data, (fp, p, T, Mw, name))
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

    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=false)
    println("n_batches = $(length(train_loader)), batch_size = $batch_size")
    # flush(stdout)
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
        # MultiHeadAttention(nout * 4; nheads=4, init=Flux.glorot_uniform),
        Dense(nout * 4, nout * 2, tanh, init=Flux.glorot_normal),
        Dense(nout * 2, nout, x -> x, init=Flux.zeros32), # Allow unbounded negative outputs; parameter values physically bounded in SAFT layer
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

# todo split into two functions; parameter generation and Vₗ, p_sat calculation
# function get_SAFT_params(model, X; b=[2.5, 3.5, 12.0, 250.0], c=Float64[1, 1, 10, 100])
function calculate_saft_parameters(model, fp, Mw; b=[2.5, 3.5, 12.0, 250.0], c=[1.0, 1, 10, 100])

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
    # m = max(1.0, m) # using a max function zeros derivatives, potentially erroneously 
    α = 2
    m = log(1 + exp(α * (m - 1))) / α + 1

    # saft_input = vcat(Mw, biased_params[1:2], [λ_a], biased_params[3:4])
    saft_input = [Mw, m, σ, λ_a, λ_r, ϵ]
    return saft_input
end

function SAFT_head(model, X)
    fp, p, T, Mw, name = X

    saft_input = calculate_saft_parameters(model, fp, Mw)

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
    else
        #* Instead of ignoring, could:
        #* - compare to critical point
        #* - compare to sat_p, Vₗ at reduced T (T/Tc^exp vs T/Tc)
        ŷ_1 = nothing
        ŷ_2 = nothing
    end

    return [ŷ_1, ŷ_2]
end

function eval_loss(X_batch, y_batch, metric, model)
    batch_loss = 0.0
    n = 0
    for (X, y_vec) in zip(X_batch, y_batch)
        ŷ_vec = SAFT_head(model, X)
        # ŷ_vec = model(X)

        for (ŷ, y) in zip(ŷ_vec, y_vec)
            if !isnothing(ŷ)
                batch_loss += metric(y, ŷ)
                n += 1
                #! This zeros the gradient, making improvement impossible
                # else 
                #     batch_loss += 1000.0 # penalise NaNs
            end
        end

    end
    # @show length(y_batch) - n/2
    if n > 0
        batch_loss /= n
    end
    # penalize batch_loss depending on how many failed
    # n_failed = length(y_batch) - n
    # batch_loss += n_failed * 1000.0
    n_failed = length(y_batch) * 2 - n
    print(" $n_failed,")
    # flush(stdout)
    return batch_loss
end

function eval_loss_par(X_batch, y_batch, metric, model, n_chunks)
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
                p[i] = eval_loss(X_chunks[i], y_chunks[i], metric, model)
            end
        end
    end
    print("\n")
    # flush(stdout)
    return sum(p) / n_chunks # average partial losses
end

function percent_error(y, ŷ)
    return 100 * abs(y - ŷ) / y
end

function mse(y, ŷ)
    return ((y - ŷ) / y)^2
end

function train_model!(model, train_loader, test_loader; epochs=10, log_filename="params_log.csv")
    optim = Flux.setup(Flux.Adam(1e-3), model) # 1e-3 usually safe starting LR
    # optim = Flux.setup(Descent(1e-3), model)

    println("training on $(Threads.nthreads()) threads")
    flush(stdout)

    # todo report time for each epoch
    for epoch in 1:epochs
        epoch_start_time = time() # Start timing the epoch

        batch_loss = 0.0
        # unique_fps = Set({})
        # unique_fps = Set{Tuple{String,Tuple{Float64...}}}()
        unique_fps = Dict()

        for (X_batch, y_batch) in train_loader
            for (fp, p, T, Mw, name) in X_batch
                if !haskey(unique_fps, name)
                    unique_fps[name] = (fp, Mw)
                end
            end

            # n_failed = 0
            loss, grads = Flux.withgradient(model) do m
                loss = eval_loss_par(X_batch, y_batch, mse, m, Threads.nthreads())
                # loss = eval_loss(X_batch, y_batch, mse, m)
                loss
            end
            batch_loss += loss
            @assert !isnan(loss)

            Flux.update!(optim, model, grads[1])
        end

        # Now process unique fingerprints after all batches for the epoch are done
        for (name, (fp, Mw)) in unique_fps
            # Assuming a function calculate_saft_parameters exists
            Mw, m, σ, λ_a, λ_r, ϵ = calculate_saft_parameters(model, fp, Mw)

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

function create_ff_model_ppcsaft(nfeatures)
    nout_ppcsaft = 3
    model = Chain(
        Dense(nfeatures, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, nout_ppcsaft, x -> x), # Allow unbounded negative outputs; parameter values physically bounded in SAFT layer
    )
    # model(x) = m, σ, ϵ
    return model
end

function main(; epochs=5000)
    train_loader, test_loader = create_data(n_points=50, batch_size=230) # Should make 5 batches / epoch. 256 / 8 gives 32 evaluations per thread
    @show n_features = length(first(train_loader)[1][1][1])

    model = create_ff_model_ppcsaft(n_features)
    model_state = load("model_state_pcpsaft.jld2", "model_state")
    Flux.loadmodel!(model, model_state)
    model = Chain(model.layers[1:end-1]..., Dense(256, 4, x -> x, init=Flux.glorot_normal))
    # Freeze weights in initial layers, re-train, unfreeze, fine-tune

    train_model!(model, train_loader, test_loader; epochs=epochs)
end

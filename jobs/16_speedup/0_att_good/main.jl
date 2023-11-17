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

    # desc = get_descriptors(mol)
    # relevant_keys = [
    #     "CrippenClogP",
    #     "NumHeavyAtoms",
    #     "amw",
    #     "FractionCSP3",
    # ]
    # relevant_desc = [desc[k] for k in relevant_keys]
    # append!(fp, relevant_desc)

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

    # if pretraining
    T = Float64
    # X_data should be a datastructure of (name -> (fp, Mw, [Tr_vec, Vₗ_sat_vec, p_sat_vec]))
    train_data = Dict{String, Tuple{Vector{T}, T, Vector{T}, Vector{Vector{T}}}}()

    for (name, smiles, Mw) in mol_data
        X_vec = Vector{Float64}()
        Y_vec = Vector{Vector{Float64}}()
        fp = make_fingerprint(smiles)

        Tr_range = range(0.5, 0.975, n_points)
        for Tr in Tr_range
            T = Tr*Tc
            (p_sat, Vₗ_sat, Vᵥ_sat) = saturation_pressure(saft_model, T)

            push!(X_vec, Tr)
            push!(Y_vec, [p_sat, Vₗ_sat])
        end
        train_data[name] = (fp, Mw, X_vec, Y_vec)
    end

    #* Remove columns from fingerprints
    #! How the f*ck do I mask out columns for this new datastructure :(
    # Identify zero & one columns
    # for num = [0, 1]
    #     num_cols = length(X_data[1][1])
    #     zero_cols = trues(num_cols)
    #     for (vec, _...) in X_data
    #         zero_cols .&= (vec .== num)
    #     end
    #     keep_cols = .!zero_cols # Create a Mask
    #     X_data = [(vec[keep_cols], vals...) for (vec, vals...) in X_data] # Apply Mask
    # end

    # train_data, test_data = splitobs((X_data, Y_data), at=1.0, shuffle=false)

    # train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=false)
    # test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=false)
    # println("n_batches = $(length(train_loader)), batch_size = $batch_size")
    # return train_loader, test_loader
    return train_data
end

function calculate_saft_parameters(model, fp, Mw)
    λ_a = 6.0
    pred_params = model(fp)

    m, σ, λ_r, ϵ = pred_params .+ 1.0

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
    (fp, Mw, name, Tr, Tc, Vc, sat_p, sat_Vl, sat_Vv) = X

    saft_input = calculate_saft_parameters(model, fp, Mw)
    # Tc = critical_temperature_NN(saft_input)
    Tc2 = Tc - ∂²A∂V²(X, Vc, Tc)/∂³A∂V²∂T(X, Vc, Tc)

    T = Tr * Tc2
    # sat_p = saturation_pressure_NN(saft_input, T)
    sat_p2 = -(eos(NN_model, sat_Vv, T) - eos(NN_model, sat_Vl, T)) / (sat_Vv - sat_Vl)

    # Vₗ = volume_NN(saft_input, sat_p, T)
    sat_Vl2 = sat_Vl - (pressure_NN(X, sat_Vl, T) - sat_p) / ∂p∂V(X, sat_Vl, T)

    ŷ_1 = !isnan(sat_Vl2) ? sat_Vl2 : nothing
    ŷ_2 = !isnan(sat_p2) ? sat_p2 : nothing

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
            fp, Tr, Mw, name = X
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

function create_Y_data(mol_dict, batch_mols)
    Y_data = Vector{Vector{Float64}}()
    for mol in batch_mols
        Y_vec = last(mol_dict[mol])
        push!(Y_data, Y_vec)
    end
    return Y_data
end

function create_X_data(mol_dict, batch_mols)
    X_temp::Vector{Any} = zeros(length(batch_mols))  # Initialize an empty vector for X data

    # todo: make this multithreaded
    #? Can use @Threads.threads, not being differentiated
    @Threads.threads for (i, mol) in enumerate(batch_mols)
        mol_data = Vector{Tuple}()
        # Extract data for the molecule
        fp, Mw, X_vec, _ = mol_dict[mol]

        saft_params = calculate_saft_parameters(model, fp, Mw)
        model = make_model(saft_params...)

        (Tc, pc, Vc) = crit_pure(model)

        # Iterate through Tr in X_vec
        for (i, Tr) in enumerate(X_vec)
            T = Tc * Tr
            if i == 1
                (p_sat, Vₗ, Vᵥ) = saturation_pressure(model, T)
            else
                (p_sat, Vₗ, Vᵥ) = saturation_pressure(model, T, x0)
            end

            x0 = [Vₗ, Vᵥ]

            push!(mol_data, (fp, Mw, mol, Tr, Tc, p_sat, Vₗ, Vᵥ))
        end
        # I think this is automagically safe under @Threads.threads
        X_temp[i] = mol_data
    end
    # Concatenate all vectors into a single vector
    X_data = vcat(X_temp...)
    return X_data
end

function train_model!(model, mol_dict, optim; epochs=1000, batch_size=400, pretraining=false)
    nthreads = pretraining ? min(10, Threads.nthreads()) : Threads.nthreads()

    log_filename = pretraining ? "params_log_pretraining.csv" : "params_log.csv" 
    open(log_filename, "a") do io
        write(io, "epoch;name;Mw;m;σ;λ_a;λ_r;ϵ\n")
    end

    println("training on $nthreads threads")
    flush(stdout)

    training_molecules = collect(keys(mol_dict))
    mol_loader = DataLoader(training_molecules, batch_size = batch_size)

    for epoch in 1:epochs
        epoch_start_time = time() # Start timing the epoch
        batch_loss = 0.0

        # Objective: process mol_data into X_train, Y_train

        # We have a dictionary
        # where
        #       train_data = Dict(name => (fp, Mw, X_vec, Y_vec))
        #       X_vec      = [Tr, ]
        #       Y_vec      = [[sat_p, Vl], ]
        # and we return
        #       X_iter    = [(fp, Mw, name, Tr, Tc_pred, sat_p_pred, Vl_pred, Vv_pred), ]
        #       Y_iter    = [(sat_p_target, Vl_target)]
        #! Y_train can be evaluated all at once, as this won't change during training
        #! X_train needs to be evaluated after every gradient update, as the "pred" values
        #! depend on the current state of the model

        # We begin by splitting our molecules into batches,
        # Within the training data creation loop:
            # First, we evaluate SAFT parameters with
            #       saft_params = calculate_saft_parameters(model, fp, Mw)
            # Then create the model with 
            #       model = make_model(saft_params...)
            # Then evaluate critical temperature using
            #       (Tc, pc, Vc) = crit_pure(model)
            # Then iterate through Tr in X_vec, evaluating sat_p from
            #       (p_sat, Vₗ, Vᵥ) = saturation_pressure(model, T, x0)
            # where x0 is the result from the previous step and T is given by
            #       T = Tc * Tr

        # The problem with this is how to split datapoints into batches such that
        # marching up the saturation envelope is well defined
        # b/c this needs to be re-evaluated after every gradient update

        #! Within the training loop, we evaluate Tc, sat_p, Vl using:
        #       sat_p = -(eos(NN_model, Vᵥ, T) - eos(NN_model, Vₗ, T)) / (Vᵥ - Vₗ)
        #       Tc    = Tc - ∂²A∂V²(X, Vc, Tc)/∂³A∂V²∂T(X, Vc, Tc)
        #       Vl    = vL - (pressure_NN(X, vL, T) - p) / ∂p∂V
        #! Where all of these functions are defined as they are in ChainRulesCore.rrule

        for batch_mols in mol_loader
            # Create X_data, Y_data from batch_mol
            Y_batch = create_Y_data(mol_dict, batch_mols)
            X_batch = create_X_data(mol_dict, batch_mols)

            loss, grads = Flux.withgradient(model) do m
                loss = eval_loss_par(X_batch, Y_batch, mse, m, nthreads, !pretraining)
                loss
            end
            batch_loss += loss
            @assert !isnan(loss)
            @assert !iszero(loss)

            for mol in training_molecules
                fp, Mw, _... = mol_dict[mol]
                Mw, m, σ, λ_a, λ_r, ϵ = calculate_saft_parameters(model, fp, Mw)

                open(log_filename, "a") do io
                    write(io, "$epoch;$name;$Mw;$m;$σ;$λ_a;$λ_r;$ϵ\n")
                end
            end
            Flux.update!(optim, model, grads[1])
        end


        batch_loss /= length(train_loader)
        epoch_duration = time() - epoch_start_time

        epoch % 1 == 0 && println("\nepoch $epoch: batch_loss = $batch_loss, time = $(epoch_duration)s")
        flush(stdout)
    end
end

function create_ff_model_with_attention(nfeatures)
    nout = 4
    attention_dim = nout * 8  # Assuming the attention layer follows the first Dense layer

    mha = MultiHeadAttention(attention_dim; nheads=2, dropout_prob=0.0)
    function attention_wrapper(x)
        x_reshape = reshape(x, attention_dim, 1, 1)
        y, α = mha(x_reshape, x_reshape, x_reshape)
        y_flat = reshape(y, :)
        return y_flat
    end

    return Chain(
        Dense(nfeatures, attention_dim, relu),
        attention_wrapper,
        Dense(attention_dim, nout*4, relu),
        Dense(nout*4, nout*2, relu),
        Dense(nout*2, nout, x -> x),
    )
end

function main(; epochs=5000)
    Random.seed!(1234)
    # model = main_pcpsaft()
    mol_dict = create_data(n_points = 50)
    # train_data = Dict{String, Tuple{Vector{T}, T, Vector{T}, Vector{Vector{T}}}}()
    @show n_features = length()
    model = create_ff_model_with_attention(n_features)

    optim = Flux.setup(Flux.Adam(1e-4), model)
    train_model!(model, mol_dict, optim; epochs=1000, batch_size=400, pretraining=false)
end
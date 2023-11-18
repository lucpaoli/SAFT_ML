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
using Base.Threads: @spawn, @threads
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

    mol_data = zip(df.common_name, df.isomeric_SMILES, df.Mw, df.expt_T_min_liberal, df.expt_T_max_liberal)
    println("Generating data for $(length(mol_data)) molecules...")

    # if pretraining
    T = Float64
    # X_data should be a datastructure of (name -> (fp, Mw, [Tr], [p_sat, Vl_sat]))
    train_data = Dict{String, Tuple{Vector{T}, T, Vector{T}, Vector{Vector{T}}}}()

    for (name, smiles, Mw, T_exp_min, T_exp_max) in mol_data
        X_vec = Vector{Float64}()
        Y_vec = Vector{Vector{Float64}}()
        fp = make_fingerprint(smiles)
        saft_model = PPCSAFT([name])

        if !pretraining
            Tc, pc, Vc = crit_pure(saft_model)

            T_min = T_exp_min
            T_max = min(T_exp_max, 0.975 * Tc)
            @assert T_min < T_max
            for T in range(T_min, T_max, n_points)
                Tr = T / Tc
                (p_sat, Vl_sat, Vv_sat) = saturation_pressure(saft_model, T)

                push!(X_vec, Tr)
                push!(Y_vec, [p_sat, Vl_sat])
            end
            train_data[name] = (fp, Mw, X_vec, Y_vec)
        else
            m = saft_model.params.segment.values[1]
            sigma = saft_model.params.sigma.values[1] * 1e10
            λ_r = 15.0
            epsilon = saft_model.params.epsilon.values[1]

            # push!(X_data, (fp, nothing, Mw, name))
            push!(Y_vec, [m, sigma, λ_r, epsilon])

            train_data[name] = (fp, Mw, [1.0], Y_vec)
        end
    end

    #* Remove useless columns from fingerprints
    # Identify zero & one columns
    for num in [0, 1]
        first_fp = first(collect(values(train_data)))[1]
        zero_cols = (first_fp .== num)

        for (name, (fp, _...)) in train_data
            zero_cols .&= (fp .== num)
        end

        keep_cols = .!zero_cols # Create a Mask

        # Apply the Mask to the fingerprints in the data structure
        train_data = Dict(name => (fp[keep_cols], Mw, Tr, psat_Vlsat) for (name, (fp, Mw, Tr, psat_Vlsat)) in train_data)
    end
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
    # (fp, Mw, Tr, Tc, Vc, sat_p, sat_Vl, sat_Vv) = X
    (fp, Mw, Tr, Tc, Vc, sat_p, sat_p_Vl, sat_Vl, sat_Vv) = X
    # (fp, Mw, mol, Tr, Tc, p_sat, Vₗ, Vᵥ)

    saft_params = calculate_saft_parameters(model, fp, Mw)
    saft_model = make_NN_model(saft_params...)

    Tc2 = Tc - ∂²A∂V²(saft_params, Vc, Tc)/∂³A∂V²∂T(saft_params, Vc, Tc)
    @show Tc, Tc2

    T = Tr * Tc2
    # sat_p = saturation_pressure_NN(saft_input, T)
    sat_p2 = -(eos(saft_model, sat_Vv, T) - eos(saft_model, sat_Vl, T)) / (sat_Vv - sat_Vl)
    @show sat_p, sat_p2

    # Vₗ = volume_NN(saft_input, sat_p, T)
    sat_Vl2 = sat_Vl - (pressure_NN(saft_params, sat_Vl, T) - sat_p_Vl) / ∂p∂V(saft_params, sat_Vl, T)
    @show sat_Vl, sat_Vl2

    ŷ_1 = !isnan(sat_p2) ? sat_p2 : nothing
    ŷ_2 = !isnan(sat_Vl2) ? sat_Vl2 : nothing

    return [ŷ_1, ŷ_2]
end

function eval_loss(X_batch, y_batch, metric, model, use_saft_head)
    batch_loss = 0.0
    n = 0
    n_failed = 0
    
    for (X, y_vec) in zip(X_batch, y_batch)
        if use_saft_head
            ŷ_vec = SAFT_head(model, X)
        else
            fp, Mw = X
            ŷ = calculate_saft_parameters(model, fp, Mw)
            ŷ_vec = [ŷ[2], ŷ[3], ŷ[5] + 4*randn(), ŷ[6]]
        end

        for (ŷ, y) in zip(ŷ_vec, y_vec)
            if !isnothing(ŷ)
                batch_loss += metric(y, ŷ)
                n += 1
            else
                n_failed += 1
            end
        end
    end
    if n > 0
        batch_loss /= n
    end
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
        append!(Y_data, Y_vec)
    end
    return Y_data
end

function create_pretraining_X_data(mol_dict, batch_mols)
    X_data = Vector{Tuple}()
    for mol in batch_mols
        fp, Mw, _... = mol_dict[mol]
        push!(X_data, (fp, Mw))
    end
    return X_data
end

function create_X_data(model, mol_dict, batch_mols)
    # X_temp::Vector{Any} = zeros(length(batch_mols))  # Initialize an empty vector for X data
    X_temp = Vector{Vector{Tuple}}(undef, length(batch_mols))

    @Threads.threads for i in 1:length(batch_mols)
    # for i in 1:length(batch_mols)
        mol = batch_mols[i]
        mol_data = Vector{Tuple}()
        # Extract data for the molecule
        fp, Mw, X_vec, _ = mol_dict[mol]

        saft_params = calculate_saft_parameters(model, fp, Mw)
        saft_model = make_model(saft_params...)

        (Tc, pc, Vc) = crit_pure(saft_model)

        x0 = nothing

        # Iterate through Tr in X_vec
        for Tr in X_vec
            T = Tc * Tr
            if isnothing(x0)
                (p_sat, Vₗ, Vᵥ) = saturation_pressure(saft_model, T)
            else
                (p_sat, Vₗ, Vᵥ) = saturation_pressure(saft_model, T, x0)
            end

            @assert Vₗ != Vᵥ "volumes equal for $saft_params at T=$T"
            @assert !isnan(p_sat) "sat solver failed for $saft_params at T=$T"
            @assert !isnan(Vₗ) "sat solver failed for $saft_params at T=$T"
            @assert !isnan(Vᵥ) "sat solver failed for $saft_params at T=$T"

            p_sat_Vl = pressure(saft_model, Vₗ, T)

            x0 = [Vₗ, Vᵥ]

            # (fp, Mw, Tr, Tc, Vc, sat_p, sat_p_Vl, sat_Vl, sat_Vv) = X
            push!(mol_data, (fp, Mw, Tr, Tc, Vc, p_sat, p_sat_Vl, Vₗ, Vᵥ))
        end
        # I think this is automagically safe under @Threads.threads
        X_temp[i] = mol_data
    end
    # Concatenate all vectors into a single vector
    X_data = vcat(X_temp...)
    return X_data
end

function train_model!(model, mol_dict, optim; epochs=1000, batch_size=8, pretraining=false)
    nthreads = pretraining ? min(10, Threads.nthreads()) : Threads.nthreads()

    log_filename = pretraining ? "params_log_pretraining.csv" : "params_log.csv" 
    open(log_filename, "a") do io
        write(io, "epoch;name;Mw;m;σ;λ_a;λ_r;ϵ\n")
    end

    println("training on $nthreads threads")
    flush(stdout)

    training_molecules = collect(keys(mol_dict))
    mol_loader = DataLoader(training_molecules; batchsize = batch_size)

    # how many points?
    for epoch in 1:epochs
        epoch_start_time = time() # Start timing the epoch
        batch_loss = 0.0

        for batch_mols in mol_loader
            # Create X_data, Y_data from batch_mol
            Y_batch = create_Y_data(mol_dict, batch_mols) #! This can be created outside of epoch loop

            if !pretraining
                #* Should I cache x0 between iterations?
                X_batch = create_X_data(model, mol_dict, batch_mols)
            else
                X_batch = create_pretraining_X_data(mol_dict, batch_mols)
            end

            loss, grads = Flux.withgradient(model) do m
                loss = eval_loss_par(X_batch, Y_batch, mse, m, nthreads, !pretraining)
                loss
            end
            batch_loss += loss
            @assert !isnan(loss)
            @assert !iszero(loss)

            Flux.update!(optim, model, grads[1])
        end

        for name in training_molecules
            fp, Mw, _... = mol_dict[name]
            Mw, m, σ, λ_a, λ_r, ϵ = calculate_saft_parameters(model, fp, Mw)

            open(log_filename, "a") do io
                write(io, "$epoch;$name;$Mw;$m;$σ;$λ_a;$λ_r;$ϵ\n")
            end
        end

        n_points = length(mol_loader) * batch_size
        batch_loss /= n_points
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

function main()
    Random.seed!(1234)

    model = main_pcpsaft()

    mol_dict = create_data(n_points = 50; pretraining=false)

    optim = Flux.setup(Flux.Adam(1e-4), model)
    train_model!(model, mol_dict, optim; epochs=1, batch_size=16, pretraining=false)
end

function main_pcpsaft()
    mol_dict = create_data(n_points = 50; pretraining=true)

    @show n_features = length(first(collect(values(mol_dict)))[1])
    model = create_ff_model_with_attention(n_features)
    optim = Flux.setup(Flux.Adam(1e-4), model)
    train_model!(model, mol_dict, optim; epochs=20, batch_size=16, pretraining=true)

    return model
end
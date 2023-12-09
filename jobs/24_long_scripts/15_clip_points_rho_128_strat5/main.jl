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

using FiniteDiff

function make_fingerprint(s::String)::Vector{Float64}
    mol = get_mol(s)
    @assert !isnothing(mol)

    fp = []

    fp_str_atom_pair = get_atom_pair_fp(mol, Dict{String,Any}("radius" => 4, "nBits" => 512))

    fp_str = fp_str_atom_pair

    append!(fp, [parse(Float64, string(c)) for c in fp_str])

    return fp
end

# Generate training set for liquid density and saturation pressure
function create_data(; batch_size=16, n_points=25, pretraining=false)
    contains_only_c(name) = all(letter -> lowercase(letter) == 'c', name)

    # Create training & validation data
    df = CSV.read("../../../pcpsaft_params/training_data.csv", DataFrame, header=1)

    #* Only alkanes
    filter!(row -> occursin("Alkane", row.family), df)

    mol_df = zip(df.common_name, df.isomeric_SMILES, df.Mw, df.expt_T_min_liberal, df.expt_T_max_conservative)
    # mol_data = zip(df.common_name, df.isomeric_SMILES, df.Mw, df.expt_T_min_liberal, df.expt_T_max_liberal)
    println("Generating data for $(length(mol_df)) molecules...")

    T = Float64
    mol_data = Dict{String,Tuple{Vector{T},T,Vector{T},Vector{Vector{T}}}}()

    for (name, smiles, Mw, T_exp_min, T_exp_max) in mol_df
        fp = make_fingerprint(smiles)
        saft_model = PPCSAFT([name])

        Tc, pc, Vc = crit_pure(saft_model)

        # T_at_psat_10_Pa = 0.0
        # try
        #     T_at_psat_10_Pa, _... = saturation_temperature(saft_model, 10.0)
        # catch e
        #     println("$name, $e")
        # end

        T_max = min(T_exp_max, 500.0, 0.95*Tc)
        # T_min = max(T_exp_min, T_at_psat_10_Pa)
        T_min = T_exp_min

        if T_max < T_min
            println("Data skipped for $name: T_min ($T_min) > T_max ($T_max)")
            continue
        end
        if !pretraining
            X_vec = Vector{Float64}()
            Y_vec = Vector{Vector{Float64}}()
            for T in range(T_min, T_max, n_points)
                (p_sat, Vl_sat, Vv_sat) = saturation_pressure(saft_model, T)
                rhol_sat = 1.0 / Vl_sat

                push!(X_vec, T)
                push!(Y_vec, [p_sat, rhol_sat])
            end
            mol_data[name] = (fp, Mw, X_vec, Y_vec)
        else
            m = saft_model.params.segment.values[1]
            sigma = saft_model.params.sigma.values[1] * 1e10
            λ_r = 25.0 + 5*randn()
            epsilon = 2*saft_model.params.epsilon.values[1]

            T_min = T_exp_min

            mol_data[name] = (fp, Mw, [T_min], [[m, sigma, λ_r, epsilon]])
        end
    end

    # Split molecules into train & validation
    all_mols = collect(keys(mol_data))
    @show all_mols
    sort!(all_mols, by = mol -> mol_data[mol][2])
    # todo interlace all_mols so kfolds takes molecules [1, 6, 11, 16, etc] when sorted by molar weight, given mol_data = Dict(mol => (fp, Mw, [1.0], Y_vec))
    interlaced_mols = [all_mols[i:5:end] for i in 1:5]
    all_mols_interlaced = vcat(interlaced_mols...)

    folds = collect(kfolds(all_mols_interlaced, k=5)) # split all_mols into 5 folds
    train_mols, val_mols = folds[5]
    train_data = Dict(name => mol_data[name] for name in train_mols)

    #* Remove zero & one columns from fps based on train data
    first_fp_length = length(first(collect(values(train_data)))[1])
    one_cols = trues(first_fp_length)
    zero_cols = trues(first_fp_length)

    for (name, (fp, _...)) in train_data
        one_cols .&= (fp .== 1)
        zero_cols .&= (fp .== 0)
    end

    keep_cols = @. !zero_cols && !one_cols  # Create a Mask

    # Apply the Mask to the fingerprints in the data structure
    mol_data = Dict(name => (fp[keep_cols], vals...) for (name, (fp, vals...)) in mol_data)

    return mol_data, train_mols, val_mols
end

function calculate_saft_parameters(model, fp, Mw)
    λ_a = 6.0
    pred_params = model(fp)
    @assert !any(isnan, pred_params) "NaNs in predicted parameters"

    m, σ, λ_r, ϵ = pred_params ./ 500.0 .+ 1.0

    f(x, c) = elu(x, 0.05) + c

    m = f(m, 1.0)
    σ = f(σ, 2.0)
    λ_r = 10.0 * f(λ_r, 1.0)
    ϵ = 100.0 * f(ϵ, 1.0)

    saft_input = [Mw, m, σ, λ_a, λ_r, ϵ]
    @assert !any(isnan, saft_input) "NaNs in saft_input ?!"

    return saft_input
end

function f_sat_p(saft_params, sat_Vv, sat_Vl, T)
    saft_model = make_NN_model(saft_params...)
    sat_p2 = -(eos(saft_model, sat_Vv, T) - eos(saft_model, sat_Vl, T)) / (sat_Vv - sat_Vl + 1e-8)
    # @assert !isnan(sat_p2) "sat_p2=$sat_p2, T=$T, denom=$(sat_Vv - sat_Vl), num=$(eos(saft_model, sat_Vv, T) - eos(saft_model, sat_Vl, T)), $(eos(saft_model, sat_Vl, T)), $(eos(saft_model, sat_Vv, T)), $sat_Vl, $sat_Vv"
    return sat_p2
end

function ChainRulesCore.rrule(::typeof(f_sat_p), saft_params, sat_Vv, sat_Vl, T)
    y = f_sat_p(saft_params, sat_Vv, sat_Vl, T)

    function f_pullback(Δy)
        ∂X1 = @thunk(ForwardDiff.gradient(X -> f_sat_p(X, sat_Vv, sat_Vl, T), saft_params) .* Δy)
        ∂X2 = @thunk(ForwardDiff.derivative(X -> f_sat_p(saft_params, X, sat_Vl, T), sat_Vv) .* Δy)
        ∂X3 = @thunk(ForwardDiff.derivative(X -> f_sat_p(saft_params, sat_Vv, X, T), sat_Vl) .* Δy)
        ∂X4 = @thunk(ForwardDiff.derivative(X -> f_sat_p(saft_params, sat_Vv, sat_Vl, X), T) .* Δy)

        return (NoTangent(), ∂X1, ∂X2, ∂X3, ∂X4)
    end

    return y, f_pullback
end

function f_sat_Vl(saft_params, sat_Vl, T, sat_p_Vl)
    ∂p∂V = ForwardDiff.derivative(V -> pressure_NN(saft_params, V, T), sat_Vl)
    sat_Vl2 = sat_Vl - (pressure_NN(saft_params, sat_Vl, T) - sat_p_Vl) / (∂p∂V + 1e-8)
    # @assert !isnan(sat_Vl2) "∂p∂V = $∂p∂V"
    return sat_Vl2
end

function ChainRulesCore.rrule(::typeof(f_sat_Vl), saft_params, sat_Vl, T, sat_p_Vl)
    y = f_sat_Vl(saft_params, sat_Vl, T, sat_p_Vl)

    function f_pullback(Δy)
        # ∂X1 = ForwardDiff.gradient(X -> f_sat_Vl(X, sat_Vl, T, sat_p_Vl), saft_params) .* Δy
        #! This is a crime
        ∂X1 = @thunk(FiniteDiff.finite_difference_gradient(X -> f_sat_Vl(X, sat_Vl, T, sat_p_Vl), saft_params) .* Δy)
        ∂X2 = @thunk(ForwardDiff.derivative(X -> f_sat_Vl(saft_params, X, T, sat_p_Vl), sat_Vl) .* Δy)
        ∂X3 = @thunk(ForwardDiff.derivative(X -> f_sat_Vl(saft_params, sat_Vl, X, sat_p_Vl), T) .* Δy)
        ∂X4 = @thunk(ForwardDiff.derivative(X -> f_sat_Vl(saft_params, sat_Vl, T, X), sat_p_Vl) .* Δy)

        return (NoTangent(), ∂X1, ∂X2, ∂X3, ∂X4)
    end

    return y, f_pullback
end

function f_Tc(saft_params, Vc, Tc)
    Tc2 = Tc - ∂²A∂V²(saft_params, Vc, Tc) / ∂³A∂V²∂T(saft_params, Vc, Tc)
    # F(saft_params) = 0
    # ∂F/∂saft_params = 1

    # Tc2 = Tc - 1e-30 * ∂³A∂V²∂T(saft_params, Vc, Tc)
    return Tc2
end

function ChainRulesCore.rrule(::typeof(f_Tc), saft_params, Vc, Tc)
    y = f_Tc(saft_params, Vc, Tc)

    function f_pullback(Δy)
        ∂X1 = @thunk(FiniteDiff.finite_difference_gradient(X -> f_Tc(X, Vc, Tc), saft_params) .* Δy)
        ∂X2 = @thunk(ForwardDiff.derivative(X -> f_Tc(saft_params, X, Tc), Vc) .* Δy)
        ∂X3 = @thunk(ForwardDiff.derivative(X -> f_Tc(saft_params, Vc, X), Tc) .* Δy)

        return (NoTangent(), ∂X1, ∂X2, ∂X3)
    end

    return y, f_pullback
end

function ∂²A∂V²(X::Vector, V, T)
    return ForwardDiff.derivative(V -> pressure_NN(X, V, T), V)
    # return FiniteDiff.finite_difference_derivative(V -> pressure_NN(X, V, T), V)
end

function ∂³A∂V²∂T(X::Vector, V, T)
    return ForwardDiff.derivative(T -> ∂²A∂V²(X, V, T), T)
    # return FiniteDiff.finite_difference_derivative(T -> ∂²A∂V²(X, V, T), T)
end

function ∂³A∂V³(X::Vector, V, T)
    return ForwardDiff.derivative(V -> ∂²A∂V²(X, V, T), V)
end

function SAFT_head(model, X)
    (fp, Mw, T, Tc, Vc, sat_p, sat_p_Vl, sat_Vl, sat_Vv) = X

    saft_params::Vector{Float64} = calculate_saft_parameters(model, fp, Mw)

    Tc2 = f_Tc(saft_params, Vc, Tc)

    if abs(Tc - Tc2) > 1e-3
        return [nothing, nothing]
    end

    sat_p2 = f_sat_p(saft_params, sat_Vv, sat_Vl, T)
    sat_Vl2 = f_sat_Vl(saft_params, sat_Vl, T, sat_p_Vl)

    ŷ_1 = !isnan(sat_p2) ? sat_p2 : nothing
    ŷ_2 = !isnan(sat_Vl2) ? 1.0 / sat_Vl2 : nothing

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
            ŷ_vec = [ŷ[2], ŷ[3], ŷ[5], ŷ[6]]
        end
        # @show y_vec, ŷ_vec
        # @assert !any(isnan.(y_vec)) "NaN in y, y_vec = $y_vec"
        # @assert !any(isnan.(ŷ_vec)) "NaN in y_hat, y_hat_vec = $ŷ_vec"

        # batch_loss += mean(metric(y_vec, ŷ_vec))
        for (y, ŷ) in zip(y_vec, ŷ_vec)
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
    # use_saft_head && print("$n_failed, ")
    use_saft_head && println("n_failed = $n_failed")
    return batch_loss
end

function eval_loss_par(X_batch, y_batch, metric, model, n_chunks, use_saft_head)
    n = length(X_batch)
    chunk_size = n ÷ n_chunks

    p = bufferfrom(zeros(n_chunks))

    # Creating views for each chunk
    X_chunks = vcat([view(X_batch, (i-1)*chunk_size+1:i*chunk_size) for i in 1:n_chunks-1], [view(X_batch, (n_chunks-1)*chunk_size+1:n)])
    y_chunks = vcat([view(y_batch, (i-1)*chunk_size+1:i*chunk_size) for i in 1:n_chunks-1], [view(y_batch, (n_chunks-1)*chunk_size+1:n)])

    use_saft_head && print("n_failed = ")
    @sync begin
        for i = 1:n_chunks
            @spawn begin
                p[i] = eval_loss(X_chunks[i], y_chunks[i], metric, model, use_saft_head)
            end
        end
    end
    use_saft_head && print("\n")
    use_saft_head && flush(stdout)
    return sum(p) / n_chunks # average partial losses
end

function percent_error(y, ŷ)
    return 100 * abs(y - ŷ) / y
end

function mse(y, ŷ)
    return ((y - ŷ) / y)^2
end

# function create_Y_data(mol_dict, batch_mols)
#     Y_data = Vector{Vector{Float64}}()
#     for mol in batch_mols
#         Y_vec = last(mol_dict[mol])
#         append!(Y_data, Y_vec)
#     end
#     return Y_data
# end

function create_pretraining_data(mol_dict, batch_mols)
    X_data = Vector{Tuple}()
    for mol in batch_mols
        fp, Mw, _... = mol_dict[mol]
        push!(X_data, (fp, Mw))
    end

    Y_data = Vector{Vector{Float64}}()
    for mol in batch_mols
        Y_vec = last(mol_dict[mol])
        append!(Y_data, Y_vec)
    end

    return X_data, Y_data
end

function contains_nan(element)
    if element isa Real
        return isnan(element)
    elseif element isa Vector || element isa Tuple
        return any(contains_nan, element)
    else
        @error "Undefined behaviour for type $(typeof(element)) in contains_nan, element=$element"
    end
end

function create_data(model, mol_dict, batch_mols, x0_cache, Tc0_cache)
    # X_temp::Vector{Any} = zeros(length(batch_mols))  # Initialize an empty vector for X data
    X_temp = Vector{Union{Nothing,Vector{Tuple}}}(nothing, length(batch_mols))
    # Y_data = Vector{Vector{Float64}}()
    Y_temp = Vector{Union{Nothing,Vector{Vector{Float64}}}}(nothing, length(batch_mols))

    n = 0
    cache_lock = ReentrantLock()
    mol_dict_lock = ReentrantLock()
    temp_lock = ReentrantLock()

    points_skipped = [0 for _ in 1:Threads.nthreads()]
    sat_solves_failed = [0 for _ in 1:Threads.nthreads()]
    crit_solves_failed = [0 for _ in 1:Threads.nthreads()]

    Threads.@threads for i in 1:length(batch_mols)
        mol = batch_mols[i]
        mol_data = Vector{Tuple}()
        mol_data_Y = Vector{Vector{Float64}}()
        # Extract data for the molecule
        lock(mol_dict_lock)
        fp, Mw, X_vec, Y_vec = mol_dict[mol]
        unlock(mol_dict_lock)

        saft_params = calculate_saft_parameters(model, fp, Mw)
        saft_model = make_model(saft_params...)
        T̄ = Clapeyron.T_scale(saft_model)

        Tc0 = Tc0_cache[mol]
        options = Clapeyron.NEqOptions(maxiter=20_000)
        if isnothing(Tc0)
            #* Must be a BigFloat to converge correctly
            Tc0 = BigFloat.(Clapeyron.x0_crit_pure(saft_model))
        end
        (Tc, pc, Vc) = crit_pure(saft_model, Tc0; options=options)

        deriv = ∂²A∂V²(saft_params, Vc, Tc)
        deriv2 = ∂³A∂V²∂T(saft_params, Vc, Tc)

        if Tc < 0
            println("Tc < 0, = $(round(Tc, digits=2)) for $mol, skipping...")
            crit_solves_failed[Threads.threadid()] += 1
            continue
        elseif abs(deriv) > 1e1
            println("abs(∂²A∂V²(Tc)) > 10, = $deriv, Tc=$(round(Tc, digits=2)) for $mol, skipping...")
            crit_solves_failed[Threads.threadid()] += 1
            continue
        elseif abs(deriv/deriv2) > 1e-1
            println("abs(∂²A∂V²/∂³A∂V²∂T) > 0.1, = $deriv, Tc=$(round(Tc, digits=2)) for $mol, skipping...")
            crit_solves_failed[Threads.threadid()] += 1
            continue
        end

        lock(cache_lock)
        Tc0_cache[mol] = BigFloat.([Tc/T̄, log10(Vc)])
        unlock(cache_lock)

        (Tc, pc, Vc) = Float64.([Tc, pc, Vc])

        lock(cache_lock)
        x0 = x0_cache[mol]
        unlock(cache_lock)

        mol_sat_solves_failed = 0
        # Iterate through Tr in X_vec
        for (j, (T, Y)) in enumerate(zip(X_vec, Y_vec))
            T_threshold = 0.95 * Tc
            if T > T_threshold
                points_skipped[Threads.threadid()] += 1
                # continue
                T = T_threshold
            end

            if isnothing(x0)
                method = ChemPotVSaturation(crit=(Tc, pc, Vc))
            else
                method = ChemPotVSaturation(vl=x0[1], vv=x0[2], crit=(Tc, pc, Vc))
            end
            (p_sat, Vₗ, Vᵥ) = saturation_pressure(saft_model, T, method)

            if Vₗ == Vᵥ
                println("Volumes equal for $mol = $saft_params at T=$T, Tc=$Tc")
                sat_solves_failed[Threads.threadid()] += 1
                x0 = nothing
            elseif isnan(p_sat)
                mol_sat_solves_failed == 0 && println("\nSat solver failed for $mol = $(round.(saft_params, digits=2)) at T=$(round(T, digits=2)), Tc=$(round(Tc, digits=2)), muting...")
                mol_sat_solves_failed += 1
                sat_solves_failed[Threads.threadid()] += 1
                x0 = nothing
            else
                p_sat_Vl = pressure(saft_model, Vₗ, T)
                x0 = [Vₗ, Vᵥ]
                push!(mol_data, (fp, Mw, T, Tc, Vc, p_sat, p_sat_Vl, Vₗ, Vᵥ))
                # what is Y?
                #* Y_vec is [[p_sat, vl_sat], etc]
                push!(mol_data_Y, Y) 
            end
            if j == 1
                lock(cache_lock)
                x0_cache[mol] = x0
                unlock(cache_lock)
            end
        end
        #! I have no freaking clue why Y_data appears to go NaN suddenly
        if length(mol_data) > 0 && all(!contains_nan(x) for x in mol_data) && all(!contains_nan(x) for x in mol_data_Y)
            lock(temp_lock)
            X_temp[i] = mol_data
            Y_temp[i] = mol_data_Y
            unlock(temp_lock)
        end
    end
    println("(points skipped, sat solves failed, crit solves failed) = $(sum(points_skipped)), $(sum(sat_solves_failed)), $(sum(crit_solves_failed))")
    flush(stdout)
    # Concatenate all vectors into a single vector
    filter!(x -> !isnothing(x), X_temp)
    filter!(x -> !isnothing(x), Y_temp)
    X_data = vcat(X_temp...)
    Y_data = vcat(Y_temp...)

    # filter!(x -> !contains_nan(x), X_data)
    # filter!(x -> !contains_nan(x), Y_data)

    # Safety checks
    @assert all(x !== undef for x in X_data)
    @assert all(x !== nothing for x in X_data)
    @assert all(!contains_nan(x) for x in X_data)

    @assert all(x !== undef for x in Y_data)
    @assert all(x !== nothing for x in Y_data)
    @assert all(!contains_nan(x) for x in Y_data)
    return X_data, Y_data
end

function train_model!(model, optim, mol_dict, train_mols, val_mols; epochs=10000, n_batches=5, pretraining=false)
    nthreads = pretraining ? min(16, Threads.nthreads()) : Threads.nthreads()

    log_filename = pretraining ? "params_log_pretraining.csv" : "params_log.csv"
    open(log_filename, "a") do io
        write(io, "epoch;name;Mw;m;σ;λ_a;λ_r;ϵ\n")
    end

    println("training on $nthreads threads")
    flush(stdout)

    batch_size = ceil(Int, length(train_mols) / n_batches)
    train_loader = DataLoader(train_mols; batchsize=batch_size)

    Tc0_cache = Dict{String,Union{Nothing,Vector{BigFloat}}}(mol => nothing for mol in train_mols)
    x0_cache = Dict{String,Union{Nothing,Vector{Float64}}}(mol => nothing for mol in train_mols)

    Tc0_val_cache = Dict{String,Union{Nothing,Vector{BigFloat}}}(mol => nothing for mol in val_mols)
    x0_val_cache = Dict{String,Union{Nothing,Vector{Float64}}}(mol => nothing for mol in val_mols)

    # how many points?
    for epoch in 1:epochs
        epoch_start_time = time() # Start timing the epoch
        batch_loss = 0.0

        for batch_mols in train_loader
            # Create X_data, Y_data from batch_mol
            # Y_batch = create_Y_data(mol_dict, batch_mols) #! This can be created outside of epoch loop

            if !pretraining
                X_batch, Y_batch = create_data(model, mol_dict, batch_mols, x0_cache, Tc0_cache)
            else
                create_data(model, mol_dict, batch_mols, x0_cache, Tc0_cache)
                X_batch, Y_batch = create_pretraining_data(mol_dict, batch_mols)
            end
            @assert length(X_batch) == length(Y_batch) "X, Y batch lengths must be equal"

            loss, grads = Flux.withgradient(model) do m
                # loss = eval_loss_par(X_batch, Y_batch, mse, m, nthreads, !pretraining)
                loss = eval_loss(X_batch, Y_batch, mse, m, !pretraining)
                loss
            end
            @show loss
            @assert !isnan(loss)

            batch_loss += loss

            Flux.update!(optim, model, grads[1])
        end

        if epoch%1 == 0
            for name in train_mols
                fp, Mw, _... = mol_dict[name]
                Mw, m, σ, λ_a, λ_r, ϵ = calculate_saft_parameters(model, fp, Mw)

                open(log_filename, "a") do io
                    write(io, "$epoch;$name;$Mw;$m;$σ;$λ_a;$λ_r;$ϵ;train\n")
                end
            end
            for name in val_mols
                fp, Mw, _... = mol_dict[name]
                Mw, m, σ, λ_a, λ_r, ϵ = calculate_saft_parameters(model, fp, Mw)

                open(log_filename, "a") do io
                    write(io, "$epoch;$name;$Mw;$m;$σ;$λ_a;$λ_r;$ϵ;val\n")
                end
            end

            # Evaluate validation set loss
            if !pretraining
                X_val, Y_val = create_data(model, mol_dict, val_mols, x0_val_cache, Tc0_val_cache)
            else
                create_data(model, mol_dict, val_mols, x0_val_cache, Tc0_val_cache)
                X_val, Y_val = create_pretraining_data(mol_dict, val_mols)
            end
            @assert length(X_val) == length(Y_val) "X, Y batch lengths must be equal"

            # val_loss = eval_loss_par(X_val, Y_val, mse, model, nthreads, !pretraining)
            val_loss = eval_loss(X_val, Y_val, mse, model, !pretraining)
            @show val_loss

            # Process & report data
            batch_loss /= length(train_loader)
            @assert !iszero(batch_loss)

            epoch_duration = time() - epoch_start_time
            println("\nepoch $epoch: batch_loss = $batch_loss, val_loss = $val_loss, time = $(epoch_duration)s")
            flush(stdout)
        end

        epoch % 5 == 0 && GC.gc()
    end
end

function create_german_nn(nfeatures)
    return Chain(
        Dense(nfeatures, 2048, selu),
        Dense(2048, 1024, selu),
        Dense(1024, 1024, selu),
        Dense(1024, 512, selu),
        Dense(512, 128, selu),
        Dense(128, 32, selu),
        Dense(32, nout, x -> x),
    )
end

function create_ff_model_with_attention(nfeatures)
    nout = 4
    attention_dim = 1024

    mha = MultiHeadAttention(attention_dim; nheads=2, dropout_prob=0.0)
    function attention_wrapper(x)
        x_reshape = reshape(x, attention_dim, 1, 1)
        y, α = mha(x_reshape, x_reshape, x_reshape)
        y_flat = reshape(y, :)
        return y_flat
    end

    return Chain(
        Dense(nfeatures, 128, selu),
        # Dense(2048, 1024, selu),
        # attention_wrapper,
        # Dense(1024, 1024, selu),
        # Dense(1024, 512, selu),
        # Dense(512, 128, selu),
        Dense(128, 32, selu),
        Dense(32, nout, x -> x),
    )
end

function main()
    Random.seed!(1234)

    model = main_pcpsaft()

    mol_dict, train_mols, val_mols = create_data(n_points=50; pretraining=false)

    optim = Flux.setup(Flux.Adam(5e-6), model)
    train_model!(model, optim, mol_dict, train_mols, val_mols)
end

function main_pcpsaft()
    mol_dict, train_mols, val_mols = create_data(n_points=50; pretraining=true)

    @show nfeatures = length(first(collect(values(mol_dict)))[1])
    model = create_ff_model_with_attention(nfeatures)
    optim = Flux.setup(Flux.Adam(2.5e-5), model)
    train_model!(model, optim, mol_dict, train_mols, val_mols; epochs=500, pretraining=true)

    return model
end
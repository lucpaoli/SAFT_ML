# import Pkg; Pkg.activate(".")

using Revise
using Clapeyron
includet("./saftvrmienn.jl")
# These are functions we're going to overload for SAFTVRMieNN
import Clapeyron: a_res, saturation_pressure, pressure

using Flux
using Plots
using ForwardDiff, DiffResults

using Zygote, ChainRulesCore
using ImplicitDifferentiation

using CSV, DataFrames
using MLUtils
using RDKitMinimalLib 
using Statistics

function make_fingerprint(s::String)::Vector{Float32}
    mol = get_mol(s)
    @assert !isnothing(mol)

    fp = []
    fp_details = Dict{String,Any}("nBits" => 512, "radius" => 4)
    fp_str = get_morgan_fp(mol, fp_details)
    append!(fp, [parse(Float32, string(c)) for c in fp_str])

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

function create_data()::Tuple{DataLoader, DataLoader}
    df = CSV.read("./pcpsaft_params/SI_pcp-saft_parameters.csv", DataFrame, header=1)
    filter!(row -> occursin("Alkane", row.family), df)
    mol_data = zip(df.common_name, df.isomeric_smiles, df.molarweight)
    @info "generating data for $(length(mol_data)) molecules"

    T = Float32
    X_data = Vector{Tuple{Vector{T},T,T}}([])
    Y_data = Vector{Vector{T}}()

    n = 30
    for (name, smiles, Mw) in mol_data
        saft_model = PPCSAFT([name])
        Tc, pc, Vc = crit_pure(saft_model)

        fp = make_fingerprint(smiles)

        T_range = range(0.5 * Tc, 0.975 * Tc, n)
        for T in T_range
            (p₀, V_vec...) = saturation_pressure(saft_model, T)
            push!(X_data, (fp, T, Mw))
            push!(Y_data, Float32[p₀])
        end
    end

    #* shuffle=true randomises observation order every iteration

    #* Remove zero columns from fingerprints
    # Identify Zero Columns
    num_cols = length(X_data[1][1])
    zero_cols = trues(num_cols)
    for (vec, _, _) in X_data
        zero_cols .&= (vec .== 0)
    end

    # Create a Mask
    keep_cols = .!zero_cols

    # Apply Mask
    X_data = [(vec[keep_cols], val1, val2) for (vec, val1, val2) in X_data]

    train_data, test_data = splitobs((X_data, Y_data), at=0.8, shuffle = true)

    train_loader = DataLoader(train_data, batchsize=32, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=32, shuffle=false)

    return train_loader, test_loader
end

function create_model(nfeatures)
    # nfeatures = length(X_data[1][1])
    nout = 5
    unbounded_model = Chain(
        Dense(nfeatures, 2048, selu),
        Dense(2048, 1024, selu),
        Dense(1024, 512, selu),
        Dense(512, 128, selu),
        Dense(128, 32, selu),
        Dense(32, nout, selu),
    )
    # model(x) = m, σ, λ_a, λ_r, ϵ


    # Add constant bias to the model output
    b = [
        3.0,
        3.5,
        7.0,
        12.5,
        250.0,
    ]
    nn_model(x) = unbounded_model(x)/100.0 .+ b

    return nn_model, unbounded_model
end

function train_model()
    opt = Flux.ADAM(1e-3)
    epochs = 100
    train_loader, test_loader = create_data()
    nfeatures = length(train_loader.dataset[1][1][1])
    nn_model, unbounded_model = create_model(nfeatures)

    loss_fn(X_batch, y_batch) = begin
        n = 0
        batch_loss = 0.0
        for (X, y) in zip(X_batch, y_batch)
            fp, T, Mw = X
            y = y[1]

            X_pred = nn_model(fp)
            X_saft = vcat(Mw, X_pred)
            Tc = critical_temperature_NN(X_saft)
            if T < Tc
                ŷ = saturation_pressure_NN(X_saft, T)
                if !isnan(ŷ)
                    n += 1
                    batch_loss += ((ŷ - y) / y)^2
                end
            end
        end
        if n != 0
            batch_loss /= n
        end
        batch_loss
    end

    @info "Beginning training..."
    epoch_percent_loss_vec = Float32[]
    loss_vec = Float32[]
    mean_loss_vec = Float32[]
    for epoch in 1:epochs
        epoch_loss_vec = Float32[]
        epoch_loss = 0.0

        for (X_batch, y_batch) in train_loader
            # @show loss_fn(X_batch, y_batch) 

            batch_loss = 0.0
            grads = Zygote.gradient(Flux.params(unbounded_model)) do
                batch_loss = loss_fn(X_batch, y_batch)
            end

            # Update model parameters
            Flux.update!(opt, Flux.params(unbounded_model), grads)

            append!(epoch_loss_vec, batch_loss)
            append!(epoch_percent_loss_vec, 100 * sqrt(batch_loss))
        end
        mean_loss = mean(epoch_loss_vec)
        mean_percent_loss = mean(epoch_percent_loss_vec)
        append!(loss_vec, epoch_loss_vec)
        append!(mean_loss_vec, mean_loss)

        if epoch in [1, 2, 3, 4, 5, 10] || epoch % 5 == 0 || epoch == epochs
            println("Epoch: $epoch, Loss: (μ=$mean_loss, σ=$(std(epoch_loss_vec))), Percent Error: $mean_percent_loss")
        end
    end


    return nn_model
end
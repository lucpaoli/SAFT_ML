import Pkg; Pkg.activate(".")

using Revise

# Equations of state
using Clapeyron
includet("./saftvrmienn.jl")
import Clapeyron: a_res

# Generating molecular feature vectors
using RDKitMinimalLib 

# Machine learning
using Flux
using ForwardDiff, DiffResults
using Zygote, ChainRulesCore

# Misc
using Statistics, Random, Plots#, Threads
using JLD2

function differentiable_saft(X::AbstractVector{T}, Vol, Temp, Mw) where {T<:Real} model = SAFTVRMieNN(
        params=SAFTVRMieNNParams(
            Mw=[Mw],
            segment=T[X[1]],
            sigma=T[X[2]] * 1e-10,
            lambda_a=T[6.0], # Fixing at 6
            lambda_r=T[X[3]],
            epsilon=T[X[4]],
            epsilon_assoc=T[],
            bondvol=T[],
        )
    )
    return a_res(model, Vol, Temp, T[1.0])
end

function ChainRulesCore.rrule(::typeof(differentiable_saft), x, V, T, Mw)
    y = differentiable_saft(x, V, T, Mw)

    function f_pullback(Δy)
        # Use ForwardDiff to compute the gradient
        #* @thunk means derivatives only computed when needed
        ∂x = @thunk(ForwardDiff.gradient(x -> differentiable_saft(x, V, T, Mw), x) .* Δy)
        ∂V = @thunk(ForwardDiff.derivative(V -> differentiable_saft(x, V, T, Mw), V) * Δy)
        ∂T = @thunk(ForwardDiff.derivative(T -> differentiable_saft(x, V, T, Mw), T) * Δy)
        ∂Mw = @thunk(ForwardDiff.derivative(Mw -> differentiable_saft(x, V, T, Mw), Mw) * Δy)
        return (NoTangent(), ∂x, ∂V, ∂T, ∂Mw)
    end

    return y, f_pullback
end

function make_fingerprint(s::String)::Vector{Float32}
    mol = get_mol(s)
    @assert !isnothing(mol)

    fp = []
    fp_details = Dict{String,Any}("nBits" => 256, "radius" => 3)
    fp_str = get_morgan_fp(mol, fp_details)
    append!(fp, [parse(Float32, string(c)) for c in fp_str])

    fp_details = Dict{String,Any}("nBits" => 128, "radius" => 1)
    fp_str = get_morgan_fp(mol, fp_details)
    append!(fp, [parse(Float32, string(c)) for c in fp_str])

    # Additional descriptors
    desc = get_descriptors(mol)
    # sort by key
    # desc = sort(collect(desc), by=x->x[1])
    relevant_keys = [
        "CrippenClogP",
        "NumHeavyAtoms",
        "amw",
        "FractionCSP3",
    ]

    # @show desc
    relevant_desc = [desc[k] for k in relevant_keys]
    append!(fp, last.(relevant_desc))

    return fp
end

function main()
    Random.seed!(1234)

    # Initially sample data for hydrocarbons
    #! isobutane, isopentane not defined for SAFTVRMie
    println("creating data")
    species = [
        "methane",
        "ethane",
        "propane",
        "butane",
        # "pentane",
        # "hexane",
        # "heptane",
        # "octane",
        # "nonane",
        # "decane",
    ]

    # Define smiles map
    smiles_map = Dict(
        "methane" => "C",
        "ethane" => "CC",
        "propane" => "CCC",
        "butane" => "CCCC",
        "isobutane" => "CC(C)C",
        "pentane" => "CCCCC",
        "isopentane" => "CC(C)CC",
        "hexane" => "CCCCCC",
        "heptane" => "CCCCCCC",
        "octane" => "CCCCCCCC",
        "nonane" => "CCCCCCCCC",
        "decane" => "CCCCCCCCCC",
    )


    # X data contains fingerprint, V, T
    # Y data contains a_res
    #* Sampling data along saturation curve
    T = Float32
    X_data = Vector{Tuple{Vector{T},T,T, String}}([])
    Y_data = Vector{Vector{T}}()

    n = 100
    for s in species
        # model = GERG2008([s])
        model = SAFTVRMie([s])
        Tc, pc, Vc = crit_pure(model)
        smiles = smiles_map[s]

        fingerprint = make_fingerprint(smiles)

        T_range = range(0.5 * Tc, 0.99 * Tc, n)
        # V_range = range(0.5 * Vc, 1.5 * Vc, n) # V could be sampled from a logspace
        for T in T_range
            (p₀, V_vec...) = saturation_pressure(model, T)
            for V in V_vec
                push!(X_data, (fingerprint, V, T, s))
                a = a_res(model, V, T, [1.0])
                push!(Y_data, Float32[a])
            end
        end
    end

    # Randomly shuffle data
    #? Split into train, validation set too?
    # Generate a set of shuffled indices
    # shuffled_indices = shuffle(1:length(X_data))
    idx = collect(range(1, length(X_data)))
    shuffle!(idx)

    # Rearrange X_data and y_data according to the shuffled indices
    X_data = X_data[idx]
    Y_data = Y_data[idx]

    # Generate nominal X dictionary
    nominal_X = Dict{String,Vector{Float32}}()
    for s in species
        model = SAFTVRMie([s])

        nominal_X[s] = [
            model.params.segment[1],
            model.params.sigma[1]*1e10,
            # model.params.lambda_a[1],
            model.params.lambda_r[1],
            model.params.epsilon[1],
        ]
    end
    @show nominal_X #! Looks like lambda_a should be fixed at 6.0

    Mw_dict = Dict{String,Float32}()
    for s in species
        model = SAFTVRMie([s])

        Mw_dict[s] = model.params.Mw[1]
    end

    println("creating model")
    # Define the unbounded neural network model
    input_dim = length(X_data[1][1])
    hidden_dim = 2048
    output_dim = 4

    # Using relu activation function for now
    num_layers = 6
    layers = vcat(
        [Dense(input_dim, hidden_dim, tanh),],
        [Dense(hidden_dim, hidden_dim, tanh) for _ in 1:num_layers],
        [Dense(hidden_dim, output_dim)],
    )

    # Scaling factor needed for sigma
    # sf = [1.0, 1e-10, 1.0, 1.0]
    # unbounded_model = x -> sf .* Chain(layers...)(x)
    unbounded_model = Chain(layers...)

    # Custom barrier function to bound the output within ±n⨯ of the nominal value while keeping derivatives
    # https://www.desmos.com/calculator/wpabjptlup
    # https://www.desmos.com/calculator/j6rjs1pmjj

    #! Bounds from ML_SAFT paper
    bounds = Vector{Tuple{Float32,Float32}}([
        (1, 2), # m
        (3.5, 4.5), # σ
        (12, 14), # λ
        (150, 280), # ϵ
    ])

    lb = first.(bounds)
    ub = last.(bounds)

    function bounded_output(x_nn_output, b=10.0)
        return @. lb + (ub - lb) * 0.5 * (tanh(1/b * (x_nn_output - lb) / (ub - lb)) + 1)
    end

    # Model with bounded output
    bounded_model(x) = bounded_output(unbounded_model(x))

    # Hyperparameters
    learning_rate = 1e-3
    epochs = 5
    batch_size = 32

    # Optimizer
    opt = ADAM(learning_rate)

    println("Beginning iterations, initial model performance:")

    fp, V, T, s = X_data[1]
    Mw = Mw_dict[s]
    y = Y_data[1]

    @show s, V, T, Mw

    Tp = Float64
    X_nom = Vector{Tp}(nominal_X[s])
    @show X_nom
    X_unbounded = Vector{Tp}(unbounded_model(fp))
    @show X_unbounded
    X_bounded = Vector{Tp}(bounded_model(fp))
    @show X_bounded
    ŷ = differentiable_saft(X_bounded, V, T, Mw)
    @show ŷ, y

    loss_vec = []
    mean_loss_vec = Float32[]
    # Training Loop
    #! I think I should be splitting my data into batches, evaluating that batch, then stepping my optimizer?
    #! Rather than evaluating each sample individually
    # Create mini-batches
    batched_data = [(X_data[i:min(i+batch_size-1, end), :], Y_data[i:min(i+batch_size-1, end)]) for i in 1:batch_size:size(X_data, 1)]
    @show size(batched_data)

    open("./epoch_details.txt", "w") do f
        write(f, "Epoch,Mean Loss,Std Dev Loss,Mean Percent Error\n")
    end

    println("training model")
    for epoch in 1:epochs
        epoch_loss_vec = Float32[]
        epoch_percent_loss_vec = Float32[]

        # for (X, y) in zip(X_data, Y_data)
        for (X_batch, y_batch) in batched_data
            batch_loss = 0.0
            loss_fn() = begin
                for (X, y) in zip(X_batch, y_batch)
                    # Split X into fingerprint, V, T
                    fp, V, T, s = X
                    Mw = Mw_dict[s]

                    y = y[1]

                    # Compute the loss
                    # Forward pass through the neural network to get SAFT parameter predictions
                    X_pred = bounded_model(fp)
                    ŷ = differentiable_saft(X_pred, V, T, Mw)
                    batch_loss += ((ŷ - y) / y)^2
                    @assert batch_loss isa Real "Loss is not a real number, got $(typeof(loss)), X_pred = $X_pred"
                    @assert !isnan(batch_loss) "Loss is NaN, X_pred = $X_pred"
                end
                batch_loss /= size(X_batch, 1)
            end
            grads = Zygote.gradient(Flux.params(unbounded_model)) do
                loss_fn()
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

        open("./epoch_details.txt", "a") do f
            write(f, "$epoch,$mean_loss,$(std(epoch_loss_vec)),$mean_percent_loss\n")
        end
        if epoch in [1, 2, 3, 4, 5, 10] || epoch % 5 == 0 || epoch == epochs
            println("Epoch: $epoch, Loss: (μ=$mean_loss, σ=$(std(epoch_loss_vec))), Percent Error: $mean_percent_loss")
        end
    end
    
    model_state = Flux.state(unbounded_model)
    jldsave("ffnn_state.jld2"; model_state)
    # load with Flux.loadmodel!(model, model_state)
    
    # Flux.saveparams!("unbounded_model.bson", unbounded_model)

    fp, V, T, s = X_data[1]
    Mw = Mw_dict[s]
    y = Y_data[1]

    @show s, V, T, Mw

    Tp = Float64
    X_nom = Vector{Tp}(nominal_X[s])
    @show X_nom
    X_unbounded = Vector{Tp}(unbounded_model(fp))
    @show X_unbounded
    X_bounded = Vector{Tp}(bounded_model(fp))
    @show X_bounded
    ŷ = differentiable_saft(X_bounded, V, T, Mw)
    @show ŷ, y
end

main()
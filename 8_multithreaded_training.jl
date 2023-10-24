using Revise
using Clapeyron
includet("./saftvrmienn.jl")
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

# Create training & validation data
df = CSV.read("./pcpsaft_params/SI_pcp-saft_parameters.csv", DataFrame, header=1)
filter!(row -> occursin("Alkane", row.family), df)
mol_data = zip(df.common_name, df.isomeric_smiles, df.molarweight)
@info "generating data for $(length(mol_data)) molecules"

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

T = Float32
X_data = Vector{Tuple{Vector{T},T,T}}([])
Y_data = Vector{Vector{T}}()

n = 15
for (name, smiles, Mw) in mol_data
    saft_model = PPCSAFT([name])
    Tc, pc, Vc = crit_pure(saft_model)

    fp = make_fingerprint(smiles)
    append!(fp, Mw)

    T_range = range(0.5 * Tc, 0.975 * Tc, n)
    for T in T_range
        (p₀, V_vec...) = saturation_pressure(saft_model, T)
        push!(X_data, (fp, T, Mw))
        push!(Y_data, Float32[p₀])
    end
end

# Shuffle all samples into a random order
# Package into data loaders
# batchsize = 10
# train_data = DataLoader((X_data, y_data), batchsize=batchsize, shuffle=true)
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

train_loader = DataLoader(train_data, batchsize=64, shuffle=true)
test_loader = DataLoader(test_data, batchsize=64, shuffle=false)

@info "n_batches = $(length(train_loader))"
# Base NN architecture from "Fitting Error vs Parameter Performance"
nfeatures = length(X_data[1][1])
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

opt = ADAM(1e-3)

# Add constant bias to the model output
b = [
    3.0,
    3.5,
    7.0,
    12.5,
    250.0,
]
nn_model(x) = unbounded_model(x)/50.0 .+ b

# Training loop
@info "Beginning training..."
epochs = 10
epoch_percent_loss_vec = Float32[]
loss_vec = Float32[]
mean_loss_vec = Float32[]
mean_percent_loss_vec = Float32[]

const n_threads = Threads.nthreads()
# mutex = Threads.Mutex()
l = Threads.SpinLock()
thread_batch_losses = Vector{Float32}(undef, n_threads)
thread_percent_errors = Vector{Float32}(undef, n_threads)

loss_fn(X_batch, y_batch) = begin
    n = 0
    batch_loss = 0.0
    percent_error = 0.0
    for (X, y_vec) in zip(X_batch, y_batch)
        fp, T, Mw = X
        y = y_vec[1]

        X_pred = nn_model(fp)
        X_saft = vcat(Mw, X_pred)
        Tc = critical_temperature_NN(X_saft)
        if T < Tc
            ŷ = saturation_pressure_NN(X_saft, T)
            if !isnan(ŷ)
                n += 1
                batch_loss += ((ŷ - y) / y)^2
                percent_error += 100 * abs((ŷ - y) / y)
            end
        end
    end
    if n != 0
        batch_loss /= n
        percent_error /= n
    end
    batch_loss, percent_error
end

for epoch in 1:epochs
    Threads.@threads for batch_idx in 1:length(train_loader)
    # for batch_idx in 1:length(train_loader)
        # Each thread computes its own X_batch and y_batch based on batch_idx
        # Initialize DataLoader iterator and skip to the batch corresponding to batch_idx
        # todo: check this works locally (can protype multithreading locally)
        data_iterator = iterate(train_loader)
        for _ in 1:batch_idx-1
            data_iterator = iterate(train_loader, data_iterator[2])
        end
        X_batch, y_batch = data_iterator[1]
        
        local_batch_loss = 0.0
        local_percent_error = 0.0
        
        # Calculate gradients in a usual way
        grads = Zygote.gradient(Flux.params(unbounded_model)) do
            batch_loss, percent_error = loss_fn(X_batch, y_batch)
            local_batch_loss = batch_loss
            local_percent_error = percent_error
            percent_error
        end

        # Update thread-local storage after gradient calculation
        thread_batch_losses[Threads.threadid()] = local_batch_loss
        thread_percent_errors[Threads.threadid()] = local_percent_error
        # thread_batch_losses[1] = local_batch_loss
        # thread_percent_errors[1] = local_percent_error

        # Lock needed here if update is not thread-safe
        lock(l)
        try
            Flux.update!(opt, Flux.params(unbounded_model), grads)
        finally
            unlock(l)
        end
    end
    
    # Aggregate results from all threads
    # epoch_loss_vec = thread_batch_losses
    # epoch_percent_loss_vec = thread_percent_errors
    mean_loss = mean(thread_batch_losses)
    std_loss = mean(thread_batch_losses)
    mean_percent_loss = mean(thread_percent_errors)
    # append!(loss_vec, thread_batch_losses)
    append!(mean_loss_vec, mean_loss)
    append!(mean_percent_loss_vec, mean_percent_loss)

    println("Epoch: $epoch, Loss: (μ=$mean_loss, σ=$std_loss), Percent Error: $mean_percent_loss")
end

model_state = Flux.state(unbounded_model);
jldsave("model_state.jld2"; model_state)
jldsave("mean_percent_loss_vec.jld2"; mean_percent_loss_vec)

plot(epoch_percent_loss_vec)
savefig("epoch_loss.png")
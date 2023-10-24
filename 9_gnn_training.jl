using Clapeyron
include("./saftvrmienn.jl")
import Clapeyron: a_res

using MolecularGraph, Graphs
using Plots

using Flux
# using Flux: onecold, onehotbatch, logitcrossentropy
using Flux: DataLoader
using GraphNeuralNetworks
using ForwardDiff, Zygote, ChainRulesCore

using MLUtils
using OneHotArrays
# using LinearAlgebra, Random, Statistics
using Statistics, Random
using CSV, DataFrames

function make_graph_from_smiles(smiles::String)
    molgraph = smilestomol(smiles)

    g = SimpleGraph(nv(molgraph))
    for e in edges(molgraph)
        add_edge!(g, e.src, e.dst)
    end

    # Should number of hydrogens be one-hot encoded?
    f(vec, enc) = hcat(map(x -> onehot(x, enc), vec)...)
    num_h = f(implicit_hydrogens(molgraph), [0, 1, 2, 3, 4])
    hybrid = f(hybridization(molgraph), [:sp, :sp2, :sp3])
    atoms = f(atom_symbol(molgraph), [:C, :O, :N])

    # Node data should be matrix (num_features, num_nodes)
    # Matrix has num_nodes columns, num_features rows
    ndata = Float32.(vcat(num_h, hybrid, atoms))

    # h(vec, enc) = hcat(map(x -> onehot(x, enc), vec)...)
    # @show bond_order(molgraph), is_rotatable(molgraph), is_aromatic(molgraph), collect(edges(molgraph))
    b_order = Float32.(f(bond_order(molgraph), [1, 2, 3]))
    # @show b_order
    # rotatable = f(is_rotatable(molgraph), [false, true])
    # edata = Matrix{Float32}(vcat(b_order, rotatable))
    # edata = Matrix{Float32}(b_order)
    edata = nothing
    
    g = GNNGraph(g, ndata = ndata, edata = edata)
    return g
end

g = make_graph_from_smiles("C")

# Create training & validation data
df = CSV.read("./pcpsaft_params/SI_pcp-saft_parameters.csv", DataFrame, header=1)
filter!(row -> occursin("Alkane", row.family), df)
mol_data = zip(df.common_name, df.isomeric_smiles, df.molarweight)
@info "generating data for $(length(mol_data)) molecules"

# Create training data, currently sampled along saturation curve
T = GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}
graphs = T[]
states = Vector{Float32}[]
species = String[] # For checking parameter similarity
Y_data = Float32[]

n = 25
# for s in all_species
for (name, smiles, Mw) in mol_data
    saft_model = PPCSAFT([name])
    Tc, pc, Vc = crit_pure(model)

    # fingerprint = make_fingerprint(smiles)
    g = make_graph_from_smiles(smiles)

    T_range = range(0.5 * Tc, 0.99 * Tc, n)
    # V_range = range(0.5 * Vc, 1.5 * Vc, n) # V could be sampled from a logspace
    for T in T_range
        (p₀, V_vec...) = saturation_pressure(model, T)
        if !any(isnan.(V_vec))
            for V in V_vec
                push!(graphs, g)
                push!(species, name)

                # Mw = model.params.Mw.values[1]
                # m = model.params.segment.values[1]
                push!(states, [V, T, Mw])

                a = a_res(model, V, T, [1.0])
                @assert !isnan(a) "a is NaN at (V,T) = ($(V),$(T)) for $name"
                push!(Y_data, a)
            end
        else
            @warn "NaN found in V_vec at T = $T for $name"
        end
    end
end

train_data, test_data = splitobs((graphs, states, species, Y_data), at = 0.8, shuffle = true) |> getobs

Random.seed!(0)
train_loader = DataLoader(train_data, batchsize = 32, shuffle = true)
test_loader = DataLoader(test_data, batchsize = 32, shuffle = false)

function create_graphattention_model(nin, ein, nh; nout=3, nhlayers=1, afunc=relu)
    GNNChain(
        GATv2Conv((nin, ein) => nh, afunc),
        [GATv2Conv(nh => nh, afunc) for _ in 1:nhlayers]...,
        GlobalPool(mean),
        Dropout(0.2),
        Dense(nh, nh),
        Dense(nh, nout),
    )
end

function eval_loss(model, data_loader, device)
    loss = 0.0
    acc = 0.0
    for (g, state, species, y) in data_loader
        g, state, y = MLUtils.batch(g) |> device, state |> device, y |> device
        X = model(g, g.ndata.x)
        for (Xᵢ, stateᵢ, yᵢ) in zip(eachcol(X), state, y)
            V, T, Mw, m = stateᵢ
            ŷ = predict_a_res(Xᵢ, V, T, Mw, m)
            loss += ((ŷ - yᵢ) / yᵢ)^2
            acc += abs((ŷ - yᵢ) / yᵢ)
            @assert loss isa Real "Loss is not a real number, got $(typeof(loss)), X_pred = $X_pred"
            @assert !isnan(loss) "Loss is NaN, X_pred = $X_pred"
        end
        loss /= length(state)
        acc /= length(state)
    end
    loss /= length(data_loader)
    acc /= length(data_loader)
    # return loss, 100 * sqrt(loss)
    return (loss = round(loss, digits = 4),
            acc = round(100 * acc, digits = 2))
end

function train!(model; epochs=50, η=1e-2, infotime=10, log_loss=false)
    # device = Flux.gpu # uncomment this for GPU training
    device = Flux.cpu
    model = model |> device
    opt = ADAM()

    function report(epoch)
        train = eval_loss(model, train_loader, device)
        test = eval_loss(model, test_loader, device)
        @info (; epoch, train, test)
    end

    epoch_loss_vec = Float32[]
    report(0)
    for epoch in 1:epochs
        epoch_loss = 0.0
        for (g, state, species, y) in train_loader
            g, state, y = MLUtils.batch(g) |> device, state |> device, y |> device

            batch_loss = 0.0
            loss_fn() = begin
                X = model(g, g.ndata.x)
                for (Xᵢ, stateᵢ, yᵢ) in zip(eachcol(X), state, y)
                    V, T, Mw, m = stateᵢ
                    ŷ = predict_a_res(Xᵢ, V, T, Mw, m)
                    batch_loss += ((ŷ - yᵢ) / yᵢ)^2
                    @assert batch_loss isa Real "Loss is not a real number, got $(typeof(loss)), X_pred = $X_pred"
                    @assert !isnan(batch_loss) "Loss is NaN, X_pred = $X_pred"
                end
                batch_loss /= length(state)
            end

            grads = Zygote.gradient(Flux.params(model)) do
                loss_fn()
            end
            epoch_loss += batch_loss
            Flux.update!(opt, Flux.params(model), grads)
        end
        epoch_loss /= length(train_loader)
        push!(epoch_loss_vec, epoch_loss)
        
        epoch % infotime == 0 && report(epoch)
    end
    return epoch_loss_vec
end

b = [
    3.0,
    3.5,
    7.0,
    12.5,
    250.0,
]
nn_model(x) = unbounded_model(x)/50.0 .+ b
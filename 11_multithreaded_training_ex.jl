# This will prompt if necessary to install everything, including CUDA:
using Flux, CUDA, Statistics, ProgressMeter

function main()
    # Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
    noisy = rand(Float32, 2, 5000)                                    # 2×1000 Matrix{Float32}
    truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}

    # Define our model, a multi-layer perceptron with one hidden layer of size 3:
    model = Chain(
        Dense(2 => 3, tanh),   # activation function inside layer
        BatchNorm(3),
        Dense(3 => 2),
        softmax) |> gpu        # move model to GPU, if available

    # The model encapsulates parameters, randomly initialised. Its initial output is:
    out1 = model(noisy |> gpu) |> cpu                                 # 2×1000 Matrix{Float32}

    # To train the model, we use batches of 64 samples, and one-hot encoding:
    target = Flux.onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
    loader = Flux.DataLoader((noisy, target) |> gpu, batchsize=512, shuffle=true);
    # 16-element DataLoader with first element: (2×64 Matrix{Float32}, 2×64 OneHotMatrix)

    optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.

    # 
    nthreads = Threads.nthreads()
    l = Threads.SpinLock()
    # @info "training on $(nthreads) threads"

    # Training loop, using the whole data set 1000 times:
    losses = []
    for epoch in 1:1000
        batch_loss = 0.0
        # for (x, y) in loader
        Threads.@threads for batch_idx in 1:length(loader)
        # for batch_idx in 1:length(loader)
            # Get correct idx
            data_iterator = iterate(loader)
            for _ in 1:batch_idx-1
                data_iterator = iterate(loader, data_iterator[2])
            end
            x, y = data_iterator[1]

            loss, grads = Flux.withgradient(model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x)
                Flux.crossentropy(y_hat, y)
            end
            batch_loss += loss
            # mutex lock
            lock(l)
            try
                Flux.update!(optim, model, grads[1])
            finally
                unlock(l)
            end
            push!(losses, loss)  # logging, outside gradient context
        end
        batch_loss /= length(loader)
        # epoch % 100 == 0 && @info "batch_loss = $batch_loss"
    end

    optim # parameters, momenta and output have all changed
    out2 = model(noisy |> gpu) |> cpu  # first row is prob. of true, second row p(false)

    mean((out2[1,:] .> 0.5) .== truth)  # accuracy 94% so far!
end
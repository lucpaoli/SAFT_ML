- Verify gradients using code below (From chatgpt)
- Confirm single parameter estimation with [1.0] -> [m, sigma, lambda_r, epsilon] works & converges as expected


```julia
function numerical_gradient(model, x, y, ϵ=1e-5)
    # Placeholder for numerical gradients
    num_grads = Flux.Zeros(gradient(model).params)
    
    # Iterate over each parameter in the model
    for (i, p) in enumerate(params(model))
        original_value = copy(p[])
        
        # Perturb parameter
        p[] .= original_value .+ ϵ
        loss_plus = loss_function(model(x), y)
        
        # Perturb parameter in the other direction
        p[] .= original_value .- ϵ
        loss_minus = loss_function(model(x), y)
        
        # Compute numerical gradient
        num_grads[i] .= (loss_plus - loss_minus) / (2ϵ)

        # Reset parameter to original value
        p[] .= original_value
    end
    return num_grads
end

function gradient_check(model, x, y, ϵ=1e-5, threshold=1e-3)
    # Compute backpropagation gradient
    loss, backprop_grads = Flux.withgradient(() -> loss_function(model(x), y), params(model))

    # Compute numerical gradient
    num_grads = numerical_gradient(model, x, y, ϵ)

    # Compare gradients
    for (bp_grad, num_grad) in zip(backprop_grads, num_grads)
        diff = norm(bp_grad - num_grad) / (norm(bp_grad) + norm(num_grad))
        if diff > threshold
            return false, diff
        end
    end
    return true, 0.0
end
```

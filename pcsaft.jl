using Clapeyron
import Clapeyron.@comps as @comps

# Defining an abstract type for this model type
abstract type PCSAFTNNModel <: SAFTModel end

# Defining the parameters used by the model
struct PCSAFTNNParam <: EoSParam
    Mw::SingleParam{Float64}
    segment::SingleParam{Float64}
    sigma::PairParam{Float64}
    epsilon::PairParam{Float64}
    epsilon_assoc::AssocParam{Float64}
    bondvol::AssocParam{Float64}
end

# Creating a model struct called PCSAFTNN, which is a sub-type of PCSAFTNNModel, and uses parameters defined in PCSAFTNNParam
@newmodel PCSAFTNN PCSAFTNNModel PCSAFTNNParam

function PCSAFTNN(components; idealmodel=BasicIdeal, userlocations=String[], ideal_userlocations=String[], verbose=false,assoc_options = AssocOptions())
  	# Obtain a Dict of parameters. We pass in custom locations through the optional parameter userlocations.
    params,sites = getparams(components; userlocations=userlocations, verbose=verbose)
  
    # For clarity, we assign the contents of the returned dict to their own variables.
    segment = params["segment"]
    k = get(params,"k",nothing) #if k is not provided, it will be not be considered
    Mw = params["Mw"]
    # Here, we modify the values of the sigma parameter first.
    params["sigma"].values .*= 1E-10
  
    # In some cases, we may not have the unlike parameters and will need to use combining rules. You can also define your own combining rules for this.
    sigma = sigma_LorentzBerthelot(params["sigma"])
    epsilon = epsilon_LorentzBerthelot(params["epsilon"], k)
  
    epsilon_assoc = params["epsilon_assoc"]
    bondvol = params["bondvol"]
  
    bondvol,epsilon_assoc = assoc_mix(bondvol,epsilon_assoc,sigma,assoc_options) #combining rules for association. if you want to perform cross-association mixing, check the AssocOptions docs

    # Now we can create the parameter struct that we have defined.
    packagedparams = PCSAFTNNParam(Mw, segment, sigma, epsilon, epsilon_assoc, bondvol)
  
    # Although optional, it's generally good practise to cite your models!
    references = ["10.1021/ie0003887", "10.1021/ie010954d"]

    # Build the model.
    model = PCSAFTNN(packagedparams, sites, idealmodel; ideal_userlocations=ideal_userlocations, references=references, verbose=verbose,assoc_options = assoc_options)
  
    # Return the PCSAFTNN object that you have just created.
    return model
end

function Clapeyron.a_res(model::PCSAFTNNModel, V, T, z)
    return @f(a_hc) + @f(a_disp) + @f(a_assoc)
end

function a_hc(model::PCSAFTNNModel, V, T, z)
    x = z/∑(z)
    m = model.params.segment.values
    m̄ = ∑(x .* m)
    return m̄*@f(a_hs) - ∑(x[i]*(m[i]-1)*log(@f(g_hs,i,i)) for i ∈ @comps)
end

function d(model::PCSAFTNNModel, V, T, z, i)
    ϵii = model.params.epsilon.values[i,i]
    σii = model.params.sigma.values[i,i]
    return σii * (1 - 0.12exp(-3ϵii/T))
end

function ζ(model::PCSAFTNNModel, V, T, z, n)
    ∑z = ∑(z)
    x = z * (one(∑z)/∑z)
    m = model.params.segment.values
    res = N_A*∑z*π/6/V * ∑((x[i]*m[i]*@f(d,i)^n for i ∈ @comps))
end

function g_hs(model::PCSAFTNNModel, V, T, z, i, j)
    di = @f(d,i)
    dj = @f(d,j)
    ζ2 = @f(ζ,2)
    ζ3 = @f(ζ,3)
    return 1/(1-ζ3) + di*dj/(di+dj)*3ζ2/(1-ζ3)^2 + (di*dj/(di+dj))^2*2ζ2^2/(1-ζ3)^3
end

function a_hs(model::PCSAFTNNModel, V, T, z)
    ζ0 = @f(ζ,0)
    ζ1 = @f(ζ,1)
    ζ2 = @f(ζ,2)
    ζ3 = @f(ζ,3)
    return 1/ζ0 * (3ζ1*ζ2/(1-ζ3) + ζ2^3/(ζ3*(1-ζ3)^2) + (ζ2^3/ζ3^2-ζ0)*log(1-ζ3))
end

# INSERT REST OF CODE
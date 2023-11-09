import Pkg
Pkg.activate("../../../.")

println("Environment activated, loading script")
flush(stdout)

include("./main.jl")

println("Running main")
flush(stdout)
@time main()

println("Script finished executing")
flush(stdout)
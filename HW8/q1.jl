using LinearAlgebra

m = parse(Int, strip(readline()))
A = [parse.(Float64, split(readline())) for i = 1:m]
A = collect(hcat(A...)')
b = parse.(Float64, split(strip(readline())))
n = length(b)
##
function polytope_volume(A, b, N = 1000000)

    bounds = vec(minimum(b ./ A, dims = 1))

    S = rand(Float64, (n, N))
    S .*= bounds
    M = count(sum((A * S) .< b, dims = 1) .== n)

    Ṽ = prod(bounds)
    V = M / N * Ṽ
end

polytope_volume(A, b)

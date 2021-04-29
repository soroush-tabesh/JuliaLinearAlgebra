using LinearAlgebra

n = parse(Int, strip(readline()))

A = [parse.(Float64, split(readline())) for i = 1:n]
A = Matrix(hcat(A...)')

X = [parse.(Float64, split(readline())) for i = 1:n]
X = Matrix(hcat(X...))
##

Xⁱ = inv(X)
X̂ = hcat([reshape(X[:, i:i] * Xⁱ[i:i, :], :) for i = 1:n]...)

Ã = reshape(A, :)
D = diagm(X̂ \ Ã)

Â = X * D * Xⁱ

for i = 1:n
    for j = 1:n
        print(round(Â[i, j],digits=6), " ")
    end
    println()
end

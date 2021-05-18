using LinearAlgebra

m = parse(Int, strip(readline()))
n = parse(Int, strip(readline()))

A = [parse.(Float64, split(readline())) for i = 1:m]
A = Matrix(hcat(A...)')
b = parse.(Float64, split(strip(readline())))
##
G = A[1:(n-1), :]' * A[1:(n-1), :]
h = A[1:(n-1), :]' * b[1:(n-1)]

X = zeros(n, m - n + 1)

for k = n:m
    G .+= A[k, :] * A[k, :]'
    h .+= b[k] * A[k, :]
    X[:, k-n+1] .= G \ h
end

X = round.(X,digits=3)

for i=1:size(X)[1]
    for j=1:size(X)[2]
        print(X[i,j]," ")
    end
    println()
end

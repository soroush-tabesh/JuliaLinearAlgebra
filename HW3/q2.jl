using LinearAlgebra

function vector_reduce(A, v)
    v = copy(v)
    for i = 1:(size(A)[2])
        v -= (A[:, i]' * v) * A[:, i]
    end
    return v
end

function gram_schmidt(A)
    A = deepcopy(A)
    for i = 1:(size(A)[2])
        A[:, i] = vector_reduce(A[:, 1:i-1], A[:, i])
        A[:, i] = normalize(A[:, i])
    end
    return A
end
##
n = parse(Int, strip(readline()))
m = parse(Int, strip(readline()))

A = [parse.(Float64, split(readline())) for i = 1:m]
A = hcat(A...)
x = parse.(Float64, split(strip(readline())))
##
Q = gram_schmidt(A)
red_x = vector_reduce(Q, x)
err = norm(red_x)
red_x /= err

for i = 1:m
    for j = 1:n
        print(Q[j, i], " ")
    end
    println()
end

if err > 1e-12
    for i = 1:n
        print(red_x[i], " ")
    end
end
println()

using LinearAlgebra, DelimitedFiles

data = readdlm("./HW8/data/Q2/data1.txt")
l = data[1:70, :]
v = data[end-50:end-1, :]
intensity = data[end, 1]
##
mids = (v+circshift(v, (-1, 0)))[1:end-1, :] ./ 2
normals = (circshift(v, (-1, 0))-v)[1:end-1, :] * [0 1; -1 0]
A = zeros((50, 70))
for i = 1:50
    for j = 1:70
        d = l[j] - mids[i]
        A[i, j] = (normals[i] ⋅ d) / (norm(d) * norm(normals[i]))
        A[i, j] = max(A[i, j], 0) / (norm(d)^2)
    end
end
##
# intensity = parse(Int, strip(readline()))
b = ones(50) .* intensity

function BVLS(A, b, l, u, max_iter = 10000, eps = 1e-8, rate = 0.01)
    m, n = size(A)
    F = []
    U = []
    L = []
    x = ones(n) * l
    iter = 0
    while iter < max_iter
        iter += 1

        w = A' * (b - A * x)
        w ./= norm(w)

        L = findall((x .< (l + eps)) .& (w .<= 0))
        U = findall((x .> (u - eps)) .& (w .>= 0))
        F = setdiff(1:n, union(U, L))
        # Kuhn-Tucker convergence test
        if all((w[F] .< eps) .& (w[F] .> -eps))
            break
        end

        w[L] .= 0
        w[U] .= 0
        a = rate
        for i in F
            if w[i] < 0
                a = min(a, (l - x[i]) / w[i])
            else
                a = min(a, (u - x[i]) / w[i])
            end
        end

        x .+= a * w

    end
    return x
end

# p = BVLS([1 -2 3; 4 5 -6; 7 8 9; -10 11 12], [15, 16, 10, 18], 0.0, 1.0)
p = BVLS(A, b, 0.0, 1.0)
##
norm(A*p-b)
##
# writedlm("A.txt", A, ',')

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

function BVLS(A, b, l, u, max_iter = 1000, eps = 1e-8)
    m, n = size(A)
    F = Set([])
    U = Set([])
    L = Set(1:n)
    x = ones(n) * l
    iter = 0
    while length(F) < n && iter < max_iter
        iter += 1

        # Kuhn-Tucker convergence test
        w = A' * (b - A * x)
        if all(w[collect(L)] .<= 0) &&
           all(w[collect(U)] .>= 0) &&
           all((w[collect(F).<eps]) & (w[collect(F).>-eps]))
            break
        end

        # L = findall((x .< (l + eps)) & (w .<= 0))
        # U = findall((x .> (u - eps)) & (w .>= 0))
        # F = setdiff(collect(1:n), union(U, L))

        sg = zeros(n)
        sg[collect(L)] .= 1
        sg[collect(U)] .= -1
        ts = findmax(t -> sg[t] * w[t], union(U, L))[2]
        pop!(L, ts, nothing)
        pop!(U, ts, nothing)
        push!(F, ts)
        println(length(F), " ", ts, " ", w[ts], " ", x[ts])

        B = collect(union(U, L))
        bp = vec(b - A[:, B] * x[B])
        Ap = A[:, collect(F)]
        z = Ap \ bp

        if all(z .> l) && all(z .< u)
            x[collect(F)] .= z
            continue
        end

        α = Inf
        for jp = 1:length(z)
            j = collect(F)[jp]
            if z[jp] > u || z[jp] < l
                if z[jp] > x[j]
                    a = (u - x[j]) / (z[jp] - x[j])
                else
                    a = (l - x[j]) / (z[jp] - x[j])
                end
                α = min(α, a)
            end
        end

        Fnew = copy(F)
        for jp = 1:length(z)
            j = collect(F)[jp]
            x[j] += α * (z[jp] - x[j])
            if x[j] >= u - eps
                pop!(Fnew, j)
                push!(U, j)
            elseif x[j] <= l + eps
                pop!(Fnew, j)
                push!(L, j)
            end
        end
        F = Fnew
    end
    return x
end

# p = BVLS([1 -2 3; 4 5 -6; 7 8 9; -10 11 12], [15, 16, 10, 18], 0.0, 1.0)
p = BVLS(A, b, 0.0, 1.0)
##
writedlm("A.txt", A, ',')

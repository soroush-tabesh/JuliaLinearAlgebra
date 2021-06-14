using LinearAlgebra, DelimitedFiles, Plots, Clustering

data = readdlm("./HW8/data/Q4/agent-food.txt")
rpos = collect(Int64, data[:, 1:2])
r = data[:, 3]
##
function ls_mat(rpos, r, P, cQ, lambda = 0.3)
    k = size(P)[2]
    Q = zeros((k, cQ))
    for j = 1:cQ
        mask = rpos[:, 2] .== j
        # println(P[rpos[mask, 1], :])
        Q[:, j] =
            [P[rpos[mask, 1], :]; Matrix(lambda * I, k, k)] \
            [r[mask]; zeros(k)]
    end
    return Q
end

function find_err(rpos, r, U, V)
    R = U * V'
    rpred = zeros(length(r))
    for i = 1:length(r)
        rpred[i] = R[rpos[i, 1], rpos[i, 2]]
    end
    return norm(r - rpred)
end

function MF_ALS(rpos, r, k = 5, max_iter = 30)
    m, n = maximum(rpos, dims = 1)
    U = rand(Float64, (m, k))
    V = zeros((n, k))
    bU = copy(U)
    bV = copy(V)
    berr = find_err(rpos, r, U, V)
    for t = 1:max_iter
        U .+= 0.01 * rand(size(U))
        V = collect(ls_mat(rpos, r, U, n)')
        V .+= 0.01 * rand(size(V))
        U = collect(ls_mat(rpos[:, [2, 1]], r, V, m)')
        err = find_err(rpos, r, U, V)
        if err < berr
            berr = err
            bU = copy(U)
            bV = copy(V)
        end
    end
    return bU, bV, berr
end

errs = []
for k = 1:20
    U, V, err = MF_ALS(rpos, r, k)
    push!(errs, err)
end
plot(errs)

##
U, V, err = MF_ALS(rpos, r, 14)
ucorr = U * U'
sortperm(ucorr[78, :], rev = true)[2:11]
##
U, V, err = MF_ALS(rpos, r, 3)
errs = []
for k = 2:20
    # dists = [norm(V[i,:]-V[j,:]) for i=1:40,j =1:40]
    # err = norm(silhouettes(kmeans(collect(V'), k),dists))
    err = kmeans(collect(V'), k).totalcost
    push!(errs, err)
end
plot(errs)
##
asg = kmeans(collect(V'), 6).assignments
for i=1:6
    println(findall(asg.==i))
end
println()

using Plots # or StatsPlots
# using GraphRecipes  # if you wish to use GraphRecipes package too

using Clustering
using CoordinateTransformations
using StaticArrays
using LinearAlgebra
using Statistics

##
function show_scatter(arr, marker = nothing, sz = nothing, rt = false)
    gr(legends = true)
    if isnothing(sz)
        gr()
    else
        gr(size = sz)
    end
    if isnothing(marker)
        pl = scatter(arr[:, 1], arr[:, 2])
        display(pl)
    else
        pl = scatter(arr[:, 1], arr[:, 2], marker_z = marker)
        display(pl)
    end
    if rt
        return pl
    end
end


function topolar(arr)
    arr = copy(arr)
    trans = PolarFromCartesian()
    for i = 1:(size(arr)[1])
        x = SVector(arr[i, 1], arr[i, 2])
        p = trans(x)
        arr[i, 1] = p.r
        arr[i, 2] = p.θ
    end
    return arr
end

##

moons2 = moons .- [0.5 0.25]
polarmoons2 = topolar(moons2)
f(x) = sin(3 * x + 1.5) / 2 + 0.25
dt = [moons[:, 1] moons[:, 2] - f.(moons[:, 1])]
dt[:, 1] ./= maximum(dt[:, 1])
dt[:, 2] ./= maximum(dt[:, 2])

clus = kmeans(collect((dt)'), 2)
##
show_scatter(moons2, clus.assignments)
show_scatter(polarmoons2, clus.assignments)
show_scatter(dt, clus.assignments)

##
tt = scatter(moons[:, 1], moons[:, 2], rt = true)
plot!(tt, f, -2, 3)
##

show_scatter(blobs)
##

function assign_centroids(X, centroids)
    d, n = size(X)
    assignments = zeros(Int64, n)
    errs = zeros(Float64, n)
    for i = 1:n
        errs[i], assignments[i] =
            findmin(mapslices(norm, centroids .- X[:, i], dims = 1)[1, :])
    end
    return assignments, errs
end

function calc_centroids(X, assignments, k)
    return hcat(
        [
            sum(X * (assignments .== i), dims = 2) /
            max(1, count(assignments .== i)) for i = 1:k
        ]...,
    )
end

function kmeans_clusterin(
    X::AbstractMatrix{<:Real},  # data matrix (d x n)
    k::Integer,
    maxiter::Integer = 100,
    tol::Real = 1.0e-6,
)
    d, n = size(X)
    assignments = zeros(Int64, n)
    bound_mx = maximum(X, dims = 2)
    bound_mn = minimum(X, dims = 2)
    centroids = rand((d, k)) .* (bound_mx - bound_mn) .+ bound_mn

    last_er = Inf
    diff_er = Inf
    t = 0
    while diff_er > tol && (t += 1) <= maxiter
        assignments, errs = assign_centroids(X, centroids)
        centroids = calc_centroids(X, assignments, k)
        erval = sum(errs)
        diff_er = last_er - erval
        last_er = erval
    end
    return assignments, last_er, t
end

##

cluss, er, t = kmeans_clusterin(collect((blobs)'), 3)
# clusst= kmeans(collect((blobs)'), 3)

show_scatter(blobs, cluss)
# show_scatter(blobs, clusst.assignments)

##
errs = [kmeans_clusterin(collect((blobs)'), i)[2] for i = 1:10]
display(plot(errs))
elbow = errs[1:end-1] - errs[2:end]
display(plot(elbow))
function first_local_minima(arr)
    n = length(arr)
    for i=2:n-1
        if arr[i] < arr[i-1] && arr[i] < arr[i+1]
            return i
        end
    end
    return findmin(arr)[2]
end

print(first_local_minima(elbow))
##
function silhouette_score(X, assignments)
    n = size(X)[2]
    k = maximum(assignments)
    if k <= 1
        return 0
    end
    res = 0
    for i = 1:n
        pt = X[:, i]
        c = count(assignments .== assignments[i])
        if c == 1
            continue
        end
        dists = mapslices(norm, X .- pt, dims = 1)
        a = sum(dists[assignments.==assignments[i]]) / (c - 1)
        b = minimum([
            sum(dists[assignments.==j]) / count(assignments .== j) for
            j = 1:k if j != assignments[i]
        ])
        res += (b - a) / max(a, b)
    end
    return res / n
end
silo = [
    silhouette_score(
        collect((blobs)'),
        kmeans_clusterin(collect((blobs)'), i)[1],
    ) for i = 1:10
]
display(plot(silo))
print(findmax(silo)[2])

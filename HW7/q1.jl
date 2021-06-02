using LinearAlgebra, DelimitedFiles, Plots, Clustering, Polynomials, Peaks

points = readdlm("HW7/data/data1.txt", ' ')
points = points[sortperm(points[:, 1]), :]
scatter(points[:, 1], points[:, 2])

##
km = kmeans(collect(points'), 3)
scatter(points[:, 1], points[:, 2], marker_z = km.assignments)

##

function linear_regression(pts)
    A = [pts[:, 1] ones((size(pts)[1], 1))]
    b = pts[:, 2]
    x = A \ b
    err = norm(b - A * x)
    return x, err
end
moving_average(vs, n) =
    [sum(@view vs[i:(i+n-1)]) / n for i = 1:(length(vs)-(n-1))]


errs = []
s1 = 100
s2 = 900
for i = (s1+5):s2
    push!(errs, linear_regression(points[s1:i, :])[2] / i)
end
# errs = moving_average(errs,10)
plot(errs)
##
plt = scatter(points[1:100, 1], points[1:100, 2])
scatter!(points[101:800, 1], points[101:800, 2])
scatter!(points[801:1000, 1], points[801:1000, 2])

##
ervs = []
for i = 1:7
    p = fit(points[:, 1], points[:, 2], i)
    erv = p.(points[:, 1]) - points[:, 2]
    erv = sum(erv .^ 2)
    push!(ervs, erv)
end
plot(ervs)

##

f = fit(points[:, 1], points[:, 2], 6)

anchors = [
    1
    argmaxima(f.(points[:, 1]))
    argminima(f.(points[:, 1]))
    length(points[:, 1])
]
anchors = anchors[:, 1]
sort!(anchors)
anchors_x = points[anchors]

scatter(points[:, 1], points[:, 2])
scatter!(points[anchors, 1], f.(points[anchors, 1]))
display(plot!(f, extrema(points[:, 1])..., label = "chert"))

##
function line_intersection(l1, l2)
    return (l2[2] - l1[2]) / (l1[1] - l2[1])
end

for i = 2:(length(anchors)-1)
    l1 = linear_regression(points[anchors[i-1]:anchors[i], :])[1]
    l2 = linear_regression(points[anchors[i]:anchors[i+1], :])[1]
    anchors_x[i] = line_intersection(l1, l2)
end

for i = 1:(length(anchors)-1)
    mpts = points[anchors[i]:anchors[i+1], :]
    psl = linear_regression(mpts)[1]
    ps(x) = psl[1] * x + psl[2]
    plot!(plt, ps, anchors_x[i], anchors_x[i+1], label = "chert", lw = 3)
end
display(plt)

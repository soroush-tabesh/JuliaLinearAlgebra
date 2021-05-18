using LinearAlgebra, DelimitedFiles, Plots

points = readdlm("HW6/data1.txt", ',')

##

function ls_circle(pts)
    mat = [2 * pts ones(size(pts)[1])]
    b = pts[:, 1] .^ 2 + pts[:, 2] .^ 2
    ans = mat \ b
    ans[3] = sqrt(ans[3] + ans[1]^2 + ans[2]^2)
    return ans
end

function get_circle_points(spec)
    theta = collect(0:0.01:2*pi)
    x = spec[3] * cos.(theta) .+ spec[1]
    y = spec[3] * sin.(theta) .+ spec[2]
    return [x y]
end

circ_spec = ls_circle(points)
circ = get_circle_points(circ_spec)

plt = scatter(points[:, 1], points[:, 2], aspect_ratio = 1)
plot!(plt, circ[:, 1], circ[:, 2])
display(plt)

using LinearAlgebra
using Printf

n = parse(Int, strip(readline()))
c = parse.(Float64, split(strip(readline())))
b = parse(Float64, strip(readline()))
p1 = parse.(Float64, split(strip(readline())))
p2 = parse.(Float64, split(strip(readline())))

##

function reflect_about_plane(c, b, p)
    if abs(b - c' * p) < 1e-6
        return p
    end
    n = size(p)[1]
    c_norm = norm(c)
    c = c / c_norm
    b = b / c_norm
    H = Matrix(1.0 * I, n + 1, n + 1)
    H[1:n, 1:n] -= 2 * c * c'
    H[1:n, end] = 2 * c * b
    return (H*[p1; 1])[1:n]
end

function print_arr_round(arr, d)
    for elem in arr
        print(round(elem, digits = d), " ")
    end
    println()
end

function line_plane_intersect(p1, p2, c, b)
    if abs(b - c' * p1) < 1e-6
        return p1
    elseif abs(b - c' * p2) < 1e-6
        return p2
    end
    v = p2 - p1
    return p1 + ((b - c' * p1) / (c' * v)) * v

end

p1_p = reflect_about_plane(c, b, p1)
print_arr_round(p1_p, 4)

p1_f = copy(p1)
p2_f = copy(p2)

if sign(c' * p1 - b) == sign(c' * p2 - b)
    global p1_f = copy(p1_p)
end

q = line_plane_intersect(p1_f, p2_f, c, b)
print_arr_round(q,4)

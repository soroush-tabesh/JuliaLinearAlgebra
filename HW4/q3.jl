using LinearAlgebra
using Printf

m = parse(Int, strip(readline()))
A = [parse.(Float64, split(readline())) for i = 1:m]
A = Matrix(hcat(A...)')
q = parse.(Float64, split(strip(readline())))
r = parse(Int, strip(readline()))

##
function low_rank_apx(M, rnk)
    U, Sd, V = svd(M)
    if size(Sd)[1] <= rnk
        return M
    end
    Sd[rnk+1:end] .= 0
    return U * Diagonal(Sd) * V'
end

function normalize_column(M)
    return M ./ sqrt.(sum(M .^ 2, dims = 1))
end

function argmax_thresh(arr, thresh)
    mx = maximum(arr)
    res = [i for i = 1:size(arr)[1] if arr[i] > (mx - thresh)]
    return res
end

function print_arr(arr)
    for elem in arr
        print(elem, ' ')
    end
    println()
end

c1 = A' * q
Ā = normalize_column(A)
q_bar = normalize(q)
c2 = Ā' * q_bar
Ā_low = low_rank_apx(Ā, r)
c3 = Ā_low' * q_bar

print_arr(argmax_thresh(c1, 0.01))
print_arr(argmax_thresh(c2, 0.01))
print_arr(argmax_thresh(c3, 0.01))

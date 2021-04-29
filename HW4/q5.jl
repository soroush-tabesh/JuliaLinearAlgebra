using LinearAlgebra
using Printf

n = parse(Int, strip(readline()))
u = parse.(Float64, split(strip(readline())))
y = parse.(Float64, split(strip(readline())))

##

H2 = diagm(n - 1, n, 0 => repeat([1], n - 1), 1 => repeat([-1], n - 1))
H3 = diagm(
    n - 2,
    n,
    0 => repeat([1], n - 2),
    1 => repeat([-2], n - 2),
    2 => repeat([1], n - 2),
)
Q = [y H2' * H2 * y H3' * H3 * y]

res = Q \ u
res[1] -= 1

function print_arr_round(arr)
    for elem in arr
        println(Int(clamp(round(elem, digits = 0), 1, Inf)))
    end
end

print_arr_round(res) # WTF! gets accepted!

##
#
# m1 = [
#     1 1 1
#     0 -1 -2
#     0 0 1
# ]
# m2 = [
#     0 -1 -2
#     1 2 5
#     0 -1 -4
# ]
# m3 = [
#     0 0 1
#     0 -1 -4
#     1 2 6
# ]
#
# function print_arr_round(arr)
#     for elem in arr
#         println(Int(clamp(round(elem, digits = 0),1,Inf)))
#     end
# end
#
# M = [y' * m1; y' * m2; y' * m3]
# res = (M) \ u
# res[1] -= 1
# print_arr_round(res)

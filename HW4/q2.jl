using LinearAlgebra
using Printf

n = parse(Int, strip(readline()))

x = parse.(Float64, split(strip(readline())))
P = [parse.(Float64, split(readline())) for i = 1:n]
P = Matrix(hcat(P...)')

##

# last_exp = Matrix(I,n,n)
# show(stdout, "text/plain", last_exp)

# P_exp = [P]
# for i = 2:n+1
# push!(P_exp, P * P_exp[i-1])
# end
#%%

last_P = I
d = zeros(Int32, n)
eps = 1e-14
for t = 1:(2*n+2)
    global last_P = last_P * P
    for i = 1:n
        if last_P[i, i] > eps
            d[i] = gcd(d[i], t)
        end
    end
end

if d == ones(Int32, n)
    egi = inv(eigvecs(P))'
    egv = eigvals(P)
    b = egi \ x
    for i = 1:(size(egv)[1])
        if norm(egv[i] - 1) > eps
            b[i] = 0
        end
    end
    xf = egi * b
    # xf = copy(x)'
    # while true
    #     global xf = xf*P
    #     if norm(xf - xf*P) < 1e-8
    #         break
    #     end
    # end
    for e in xf
        @printf("%.7f\n", real(e))
    end
else
    for e in d
        @printf("%d\n", e)
    end
end

using LinearAlgebra, DelimitedFiles, Random, Plots

X = readdlm("HW7/data/data3.txt")
sfl = shuffle(1:size(X)[1])
y = Int.(X[sfl, end])
X = X[sfl, 1:end-1]
R = randn((13, 100))
X = X * R

##
function get_slice(arr, k, i)
    slr = zeros(Bool, (size(arr)[1]))
    slr[(size(arr)[1]÷k*(i-1)+1):(size(arr)[1]÷k*i)] .= 1
    return arr[slr, :], arr[(!).(slr), :]
end

X_test, X_train = get_slice(X, 5, 5)
y_test, y_train = get_slice(y, 5, 5)
##

function my_model(X_tr, y_tr, X_vld, n = 10)
    n = min(n,size(X_tr)[2])
    msk = y_tr .== 1
    Xp, Xn = X_tr[msk, :], X_tr[(!).(msk), :]
    Up = eigvecs(Xp' * Xp)[:, 1:n]
    Up = Up * (Up')
    Un = eigvecs(Xn' * Xn)[:, 1:n]
    Un = Un * (Un')
    pred = ones(Int, size(X_vld)[1])
    for i = 1:size(X_vld)[1]
        x = X_vld[i, :]
        dp = norm(x - Up * x)
        dn = norm(x - Un * x)
        if dp > dn
            pred[i] = -1
        end
    end
    return pred
end

res = []
for i = 1:20
    accs = []
    for fi = 1:5
        X_vld, X_tr = get_slice(X_train, 5, fi)
        y_vld, y_tr = get_slice(y_train, 5, fi)
        y_vld, y_tr = vec(y_vld), vec(y_tr)

        ŷ_vld = my_model(X_tr, y_tr, X_vld, i)
        acc = count((ŷ_vld - y_vld) .== 0) / length(y_vld)

        push!(accs, acc)
    end
    push!(res, sum(accs) / 5)
end
plot(res)
##
n = argmin(res)
ŷ_test = my_model(X_train, vec(y_train), X_test, n)
acc = count((ŷ_test - y_test) .== 0) / length(y_test)
println("best n is $n")
println("Accuracy for test data is $acc")

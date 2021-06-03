using LinearAlgebra, DelimitedFiles, Plots, MLJ, LIBSVM, Printf

data = readdlm("HW7/data/data5.txt")
sfl = shuffle(1:size(data)[1])
lbl = Int.(data[sfl, end])
data = data[sfl, 1:end-1]
##
function get_slice(arr, k, i)
    slr = zeros(Bool, (size(arr)[1]))
    slr[(size(arr)[1]÷k*(i-1)+1):(size(arr)[1]÷k*i)] .= 1
    return arr[slr, :], arr[(!).(slr), :]
end

test_data, train_data = get_slice(data, 5, 5)
test_lbl, train_lbl = get_slice(lbl, 5, 5)

##

function my_model(tr_data, tr_lbl, vld_data)
    # train
    models = []
    for m = 1:3
        m_lbl = 1 * (tr_lbl[:] .== m)
        # model = svmtrain(collect(tr_data'), m_lbl, kernel=Kernel.Linear)
        model = svmtrain(
            collect(tr_data'),
            m_lbl,
            kernel = Kernel.RadialBasis,
            gamma = 0.001,
            cost = 10.0,
        )
        push!(models, model)
    end
    #validate
    vld_pred = ones(Int, size(vld_data)[1])
    vld_dec = zeros(size(vld_data)[1])
    for m = 1:3
        pred, dec = svmpredict(models[m], collect(vld_data'))
        dec = abs.(dec)[1, :]
        msk = (pred .> 0) .& (dec .> vld_dec)
        vld_dec[msk] = dec[msk]
        vld_pred[msk] = pred[msk] * m
    end
    return vld_pred
end

function confmat_m(ŷ, y)
    return confmat(
        coerce(vec(ŷ), OrderedFactor),
        coerce(vec(y), OrderedFactor),
    )
end

function accuracy_m(ŷ, y)
    return accuracy(
        coerce(vec(ŷ), OrderedFactor),
        coerce(vec(y), OrderedFactor),
    )
end

accs = []
for fi = 1:5
    vld_data, tr_data = get_slice(train_data, 5, fi)
    vld_lbl, tr_lbl = get_slice(train_lbl, 5, fi)

    vld_pred = my_model(tr_data, tr_lbl, vld_data)
    acc = accuracy_m(vld_pred, vld_lbl) * 100

    push!(accs, acc)
    println("Fold-$fi Accuracy: $acc")
end
println("Avg Accuracy: ", sum(accs) / 5)

##

test_pred = my_model(train_data, train_lbl, test_data)
show(stdout, "text/plain", confmat_m(test_pred, test_lbl))
println(accuracy_m(test_pred, test_lbl) * 100)

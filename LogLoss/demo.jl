include("RealRealHighDimension.jl")

using DelimitedFiles
using DataFrames
# (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("LogLoss/datasets/ECG_train.txt", 
# "LogLoss/datasets/ECG_val.txt", "LogLoss/datasets/ECG_test.txt")
# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TRAIN.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TEST.csv", ',')
# y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TRAIN_labels.csv", ',')
# y_train = Int.(vec(y_train))
# y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TEST_labels.csv", ',')
# y_test = Int.(vec(y_test))

# class_A = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/guassian_curves_500_20_2_class_A.csv", ',')
# class_A_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/guassian_curves_500_20_2_class_A_labels.csv", ',')
# class_B = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/guassian_curves_500_20_2_class_B.csv", ',')
# class_B_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/guassian_curves_500_20_2_class_B_labels.csv", ',')

class_A = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_20_1_0.1_class_A.csv", ',')
class_A_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_20_1_0.1_class_A_labels.csv", ',')
class_B = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_20_1_0.1_class_B.csv", ',')
class_B_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_20_1_0.1_class_B_labels.csv", ',')

# class_A = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_10_1_0.1_class_A.csv", ',')
# class_A_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_10_1_0.1_class_A_labels.csv", ',')
# class_B = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_10_1_0.1_class_B.csv", ',')
# class_B_labels = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/guassian_curves_500_10_1_0.1_class_B_labels.csv", ',')

# file_path_train = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/BirdChicken_TRAIN.txt"
# file_path_test = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/datasets/BirdChicken_TEST.txt"

# Read and parse the data
# parsed_data_train = read_and_parse(file_path_train)
# parsed_data_test = read_and_parse(file_path_test)

# # Convert the list of lists into a DataFrame
# training_data = DataFrame(parsed_data_train, :auto)
# testing_data = DataFrame(parsed_data_test, :auto)

# training_data_matrix = Matrix{Float64}(training_data[2:end, :])
# training_data_matrix = Matrix(transpose(training_data_matrix))

# testing_data_matrix = Matrix{Float64}(testing_data[2:end, :])
# testing_data_matrix = Matrix(transpose(testing_data_matrix))

# training_labels = Int.(Vector(training_data[1, :]))
# testing_labels = Int.(Vector(testing_data[1, :]))

# training_labels = training_labels .- 1
# testing_labels = testing_labels .- 1


X_train = vcat(class_A[1:250, :], class_B[1:250, :])
X_test = vcat(class_A[251:end, :], class_B[251:end, :])
y_train = vcat(class_A_labels[1:250], class_B_labels[1:250])
y_train = Int.(vec(y_train))
y_test = vcat(class_A_labels[251:end], class_B_labels[251:end])
y_test = Int.(vec(y_test))

X_train = hcat([circshift(X_train[i, :], 10) for i in 1:size(X_train, 1)]...)
X_train = Matrix(transpose(X_train))
X_test = hcat([circshift(X_test[i, :], 10) for i in 1:size(X_test, 1)]...)
X_test = Matrix(transpose(X_test))

setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
#
encoding = fourier()
encode_classes_separately = false
train_classes_separately = false

#encoding = Basis("Legendre")
dtype = encoding.iscomplex ? ComplexF64 : Float64

opts=Options(; nsweeps=40, chi_max=2,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.05, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_random", random_walk_seed = 69)


if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=4567, chi_init=4, opts=opts, test_run=false)

    print_opts(opts)
    summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    sweep_summary(info)
end

# train = []
# test = []
# seeds = [11, 12, 15]
# for seed in seeds
#     W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)
#     push!(train, info["train_acc"])
#     push!(test, info["test_acc"])
# end

# X_train = hcat([circshift(X_train[i, :], 10) for i in 1:size(X_train, 1)]...)
# X_train_global = Matrix(transpose(X_train))
# X_test = hcat([circshift(X_test[i, :], 10) for i in 1:size(X_test, 1)]...)
# X_test_global = Matrix(transpose(X_test))
# y_train_global = y_train
# y_test_global = y_test

# train_accuracies_matrix = zeros(20, 20)
# test_accuracies_matrix = zeros(20, 20)
# for seed = 1:20
#     for j = 1:20
#         println(j)
#         X_train = hcat([circshift(X_train_global[i, :], -j) for i in 1:size(X_train_global, 1)]...)
#         X_train = Matrix(transpose(X_train))
#         X_test = hcat([circshift(X_test_global[i, :], -j) for i in 1:size(X_test_global, 1)]...)
#         X_test = Matrix(transpose(X_test))
#         y_train = y_train_global
#         y_test = y_test_global

#         W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test; random_state=seed, chi_init=4, opts=opts, test_run=false)

#         summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=false);
#         #index = find_stable_accuracy(info["train_acc"], 0.1)
#         train_accuracies_matrix[seed, j] = info["train_acc"][end]
#         test_accuracies_matrix[seed, j] = info["test_acc"][end]
#         # push!(train_accuracies_OBC, info["train_acc"][end])
#         # push!(test_accuracies_OBC, info["test_acc"][end])
#     end
# end
# opts=Options(; nsweeps=15, chi_max=20,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
# bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.05, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
# encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, algorithm = "PBC_left")

# train_accuracies_PBC_left = []
# test_accuracies_PBC_left = []
# for j = 1:20
#     println(j)
#     X_train = hcat([circshift(X_train_global[i, :], -j) for i in 1:size(X_train_global, 1)]...)
#     X_train = Matrix(transpose(X_train))
#     X_test = hcat([circshift(X_test_global[i, :], -j) for i in 1:size(X_test_global, 1)]...)
#     X_test = Matrix(transpose(X_test))
#     y_train = y_train_global
#     y_test = y_test_global

#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test; random_state=4756, chi_init=4, opts=opts, test_run=false)

#     summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=false);
#     #index = find_stable_accuracy(info["train_acc"], 0.1)
#     push!(train_accuracies_PBC_left, info["train_acc"][end])
#     push!(test_accuracies_PBC_left, info["test_acc"][end])
# end
# #medical training distribution: [35, 15, 25, 16, 10, 7, 18, 6, 46, 203]
# #circshift with samples
# opts=Options(; nsweeps=5, chi_max=30,  update_iters=1, verbosity=verbosity, dtype=Rdtype, lg_iter=KLD_iter,
# bbopt=BBOpt("CustomGD"), track_cost=false, eta=0.1, rescale = [false, true], d=2, encoding=Encoding("Stoudenmire"))
# num_runs = 10
# num_angles = 32
# train_accuracies_stoud = zeros(num_runs, num_angles)
# test_accuracies_stoud = zeros(num_runs, num_angles)
# conf_matrices_stoud = []
# for num = 1:num_runs
#     cur_conf_mat_list = []
#     X_train, y_train, X_test, y_test, test_indices = split_data(data, labels, [10, 10], 
#     [0, 1])
#     y_train_global = y_train
#     y_test_global = y_test
#     for j = 0:num_angles-1
#         #Apply circshift to each row using a comprehension and then reconstruct the matrix
#         X_train = hcat([circshift(X_train[i, :], -16*j) for i in 1:size(X_train, 1)]...)
#         # To transpose the result back to the original orientation
#         X_train = Matrix(transpose(X_train))
#         #println(X_train)

#         X_test = hcat([circshift(X_test[i, :], -16*j) for i in 1:size(X_test, 1)]...)
#         X_test = Matrix(transpose(X_test))

#         #y_train = circshift(y_train, -j)
#         #y_test = circshift(y_test, -j)

#         # X_train_global = X_train
#         # X_test_global = X_test

#         W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=456, chi_init=4, opts=opts)
#         summary = get_training_summary(W, train_states, test_states; print_stats=false)
#         cur_train_acc = info["train_acc"][end]
#         cur_test_acc = info["test_acc"][end]
#         cur_conf_mat = summary[:confmat]
#         train_accuracies_stoud[num, j+1] = cur_train_acc
#         test_accuracies_stoud[num, j+1] = cur_test_acc
#         push!(cur_conf_mat_list, cur_conf_mat)
#     end
#     push!(conf_matrices_stoud, cur_conf_mat_list)
# end

# train_accs = []
# test_accs = []
# for j = 0:73
#     X_train = training_data_matrix
#     X_test = testing_data_matrix
#     y_train = training_labels
#     y_test = testing_labels

#     X_train = hcat([circshift(X_train[i, :], 7*j) for i in 1:size(X_train, 1)]...)
#     X_train = Matrix(transpose(X_train))
#     X_test = hcat([circshift(X_test[i, :], 7*j) for i in 1:size(X_test, 1)]...)
#     X_test = Matrix(transpose(X_test))
#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test; random_state=4567, chi_init=4, opts=opts, test_run=false)
#     push!(train_accs, info["train_acc"][end])
#     push!(test_accs, info["test_acc"][end])
# end
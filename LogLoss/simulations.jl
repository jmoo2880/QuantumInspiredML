using DataFrames
using Statistics
using DelimitedFiles
include("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/RealRealFast_generic.jl")

# file_path_train = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/ArrowHead_TRAIN.txt"
# file_path_test = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/ArrowHead_TEST.txt"
# file_path_train = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/SwedishLeaf_TRAIN.txt"
# file_path_test = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/SwedishLeaf_TEST.txt"
# file_path_train = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/Fish_TRAIN.txt"
# file_path_test = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/Fish_TEST.txt"
# file_path_train = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/OSULeaf_TRAIN.txt"
# file_path_test = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/OSULeaf_TEST.txt"
# file_path_train = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/BirdChicken_TRAIN.txt"
# file_path_test = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/BirdChicken_TEST.txt"
# file_path_train = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/MedicalImages_TRAIN.txt"
# file_path_test = "/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/MedicalImages_TEST.txt"


# # Read and parse the data
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


# data = vcat(training_data_matrix, testing_data_matrix)
# labels = vcat(training_labels, testing_labels)
# labels = labels .- 1

# training_indices_to_keep = findall(x -> x != 0, training_labels)
# testing_indices_to_keep = findall(x -> x != 0, testing_labels)
# training_labels = training_labels[training_indices_to_keep]
# testing_labels = testing_labels[testing_indices_to_keep]
# training_data_matrix = training_data_matrix[training_indices_to_keep, :]
# testing_data_matrix = testing_data_matrix[testing_indices_to_keep, :]
# training_labels = training_labels .- 1
# testing_labels = testing_labels .- 1

# X_train, X_test, y_train, y_test = extract_class_samples(training_data_matrix, testing_data_matrix, training_labels, 
#                                         testing_labels, 6, 9)
# #column_indices = 1:7:512
#X_train = training_data_matrix
# # #indices = vcat([1, 2], collect(5:3:128))
# # #X_train = X_train[:, indices]
# # #X_train = X_train[:, column_indices]
#X_test = testing_data_matrix
# # #indices = vcat([1, 2], collect(5:3:128))
# # #X_test = X_test[:, indices]
# # #X_test = [[row[1]; row[2]; row[5:3:end]] for row in eachrow(X_test)]
# # #X_test = X_test[:, column_indices]
#y_train = training_labels
#y_test = testing_labels;

X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TRAIN.csv", ',')
X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TEST.csv", ',')
y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TRAIN_labels.csv", ',')
y_train = Int.(vec(y_train))
y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TEST_labels.csv", ',')
y_test = Int.(vec(y_test))

# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_train_20_40_80_40_80_1.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_test_20_40_80_40_80_1.csv", ',')
# y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_train_labels_20_40_80_40_80_1.csv", ',')
# y_train = Int.(vec(y_train))
# y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_test_labels_20_40_80_40_80_1.csv", ',')
# y_test = Int.(vec(y_test))
#matrix1 = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/fish_training.csv", ',')
# verbosity = 0
# setprecision(BigFloat, 128)
# Rdtype = Float64
# opts=Options(; nsweeps=10, chi_max=20,  update_iters=1, verbosity=verbosity, dtype=Rdtype, lg_iter=KLD_iter,
# bbopt=BBOpt("CustomGD"), track_cost=false, eta=0.1, rescale = [false, true], d=2, encoding=Encoding("Stoudenmire"))
# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_shifted_train_20_40_80_40_80_0.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_shifted_test_20_40_80_40_80_0.csv", ',')
# y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_shifted_train_labels_20_40_80_40_80_0.csv", ',')
# y_train = Int.(vec(y_train))
# y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_shifted_test_labels_20_40_80_40_80_0.csv", ',')
#y_test = Int.(vec(y_test))


verbosity = 0
setprecision(BigFloat, 128)
Rdtype = Float64
encoding = "Sahand"
chi_max = 10
eta = 0.1
nsweeps = 50
opts=Options(; nsweeps=nsweeps, chi_max=chi_max,  update_iters=1, verbosity=verbosity, dtype=Complex{Rdtype}, lg_iter=KLD_iter,
bbopt=BBOpt("CustomGD"), track_cost=false, eta=eta, rescale = [false, true], d=2, encoding=Encoding(encoding))

W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=456, chi_init=4, opts=opts);

summary = get_training_summary(W, train_states, test_states; print_stats=true)




# train_accuracies = []
# test_accuracies = []
# for seed = 1:100
#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=seed, chi_init=4, opts=opts);
#     index = find_stable_accuracy(info["train_acc"], 2.5)
#     push!(train_accuracies, info["train_acc"][index])
#     push!(test_accuracies, info["test_acc"][index])
# end

# train_accuracies = hcat(train_accuracies...)'
# train_accuracies = Matrix{Float64}(train_accuracies)
# test_accuracies = hcat(test_accuracies...)'
# test_accuracies = Matrix{Float64}(test_accuracies)
# mean_train = mean(train_accuracies, dims = 1)
# mean_train = vec(mean_train)
# mean_test = mean(test_accuracies, dims = 1)
# mean_test = vec(mean_test)
# std_train = std(train_accuracies, dims = 1)
# std_test = std(test_accuracies, dims = 1)

# x = range(0, stop = nsweeps, length = nsweeps+1)
# plot(x, mean_train, ribbon=std_train, linewidth = 4, label = "Mean Training Accuracy ± 1 SD")
# plot!(x, mean_test, ribbon=std_test, linewidth = 4, label = "Mean Testing Accuracy ± 1 SD")
# xlabel!("Sweeps")
# ylabel!("Accuracy")
# title!("Bird/Chicken - OBC - \$\\chi_{max}=$chi_max\$ - \$\\eta=$eta\$ \n $encoding Encoding With Random MPS Initialisation", titlefontsize=13)
# savefig("bird_chicken_chi=$(chi_max)_$(nsweeps)sweeps_OBC_$(encoding)_random_MPS_initialisation.pdf")
# savefig("bird_chicken_chi=$(chi_max)_$(nsweeps)sweeps_OBC_$(encoding)_random_MPS_initialisation.png")

# train_accuracies = []
# test_accuracies = []
# conf_matrices = []
# for p = 1
#     #column_indices = 1:7:512
#     X_train = training_data_matrix
#     #X_train = X_train[:, column_indices]
#     X_test = testing_data_matrix
#     #X_test = X_test[:, column_indices]
#     y_train = training_labels
#     y_test = testing_labels
#     verbosity = 0
#     setprecision(BigFloat, 128)
#     Rdtype = Float64
#     opts=Options(; nsweeps=5, chi_max=13,  update_iters=1, verbosity=verbosity, dtype=Complex{Rdtype}, lg_iter=KLD_iter,
#     bbopt=BBOpt("CustomGD"), track_cost=false, eta=0.1, rescale = [false, true], d=2, encoding=Encoding("Sahand"))
#     for j = 0
#         #Apply circshift to each row using a comprehension and then reconstruct the matrix
#         X_train = hcat([circshift(X_train[i, :], -j) for i in 1:size(X_train, 1)]...)
#         # To transpose the result back to the original orientation
#         X_train = Matrix(transpose(X_train))
#         #println(X_train)

#         X_test = hcat([circshift(X_test[i, :], -j) for i in 1:size(X_test, 1)]...)
#         X_test = Matrix(transpose(X_test))

#         #y_train = circshift(y_train, -j)
#         #y_test = circshift(y_test, -j)

#         X_train_global = X_train
#         X_test_global = X_test

#         W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=456, chi_init=4, opts=opts)
#         cur_train_acc = info["train_acc"][end]
#         cur_test_acc = info["test_acc"][end]
#         push!(train_accuracies, cur_train_acc)
#         push!(test_accuracies, cur_test_acc)
#         summary = get_training_summary(W, train_states, test_states; print_stats=true)
#         push!(conf_matrices, summary[:confmat])
#         println(j)
#     end
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

# opts=Options(; nsweeps=5, chi_max=30,  update_iters=1, verbosity=verbosity, dtype=Complex{Rdtype}, lg_iter=KLD_iter,
# bbopt=BBOpt("CustomGD"), track_cost=false, eta=0.1, rescale = [false, true], d=2, encoding=Encoding("Sahand"))
# num_runs = 10
# num_angles = 32
# train_accuracies_sahand = zeros(num_runs, num_angles)
# test_accuracies_sahand = zeros(num_runs, num_angles)
# conf_matrices_sahand = []
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
#         train_accuracies_sahand[num, j+1] = cur_train_acc
#         test_accuracies_sahand[num, j+1] = cur_test_acc
#         push!(cur_conf_mat_list, cur_conf_mat)
#     end
#     push!(conf_matrices_sahand, cur_conf_mat_list)
# end
using DataFrames
include("/Users/angusrutherford/Desktop/Honours/Project/Code/QuantumInspiredML/LogLoss/main_code.jl")

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
# training_labels = training_labels .- 1
# testing_labels = testing_labels .- 1
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
X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TRAIN.csv", ',')
X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TEST.csv", ',')
y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TRAIN_labels.csv", ',')
y_train = Int.(vec(y_train))
y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/swedish_leaf_6_9_shifted_TEST_labels.csv", ',')
y_test = Int.(vec(y_test))
#column_indices = 1:7:512
# X_train = training_data_matrix
# # indices = vcat([1, 2], collect(5:3:128))
# # X_train = X_train[:, indices]
# #X_train = X_train[:, column_indices]
# X_test = testing_data_matrix
# # indices = vcat([1, 2], collect(5:3:128))
# # X_test = X_test[:, indices]
# #X_test = [[row[1]; row[2]; row[5:3:end]] for row in eachrow(X_test)]
# #X_test = X_test[:, column_indices]
# y_train = training_labels
# y_test = testing_labels;
#shifted, no noise
# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_shifted_train_20_40_80_40_80_0.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_shifted_test_20_40_80_40_80_0.csv", ',')
# y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_shifted_train_labels_20_40_80_40_80_0.csv", ',')
# y_train = Int.(vec(y_train))
# y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_shifted_test_labels_20_40_80_40_80_0.csv", ',')
# y_test = Int.(vec(y_test))

# #noise, no shift
# X_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_train_20_40_80_40_80_1.csv", ',')
# X_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_test_20_40_80_40_80_1.csv", ',')
# y_train = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_train_labels_20_40_80_40_80_1.csv", ',')
# y_train = Int.(vec(y_train))
# y_test = readdlm("/Users/angusrutherford/Desktop/Honours/Project/Code/iTensor/sine_circle_test_labels_20_40_80_40_80_1.csv", ',')
# y_test = Int.(vec(y_test))

# left_only = false
verbosity = 0
setprecision(BigFloat, 128)
Rdtype = Float64
encoding = "Sahand"
chi_max = 10
eta = 0.1
nsweeps = 40
opts=Options(; nsweeps=nsweeps, chi_max=chi_max,  update_iters=1, verbosity=verbosity, dtype=Complex{Rdtype}, lg_iter=KLD_iter,
bbopt=BBOpt("CustomGD"), track_cost=false, eta=eta, rescale = [false, true], d=2, encoding=Encoding(encoding))
# W, info, train_states, test_states, test_lists = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=456, chi_init=4, opts=opts, algorithm = "both_one");

# summary = get_training_summary(W, train_states, test_states; print_stats=true)
# left_only = false
# if left_only == true
#     title = "Left"
#     savetitle = "left"
# else
#     title = "Right"
#     savetitle = "right"
# end

algorithm = "left"
train_accuracies_left_sahand = []
test_accuracies_left_sahand = []
# for seed = 1:2
#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=seed, chi_init=4, opts=opts, algorithm = algorithm);
#     index = find_stable_accuracy(info["train_acc"], 3.0)
#     println(index)
#     push!(train_accuracies_left_sahand, info["train_acc"][index])
#     push!(test_accuracies_left_sahand, info["test_acc"][index])
#end
W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=1, chi_init=4, opts=opts, algorithm = algorithm);
index = find_stable_accuracy(info["train_acc"], 1.0)
println(index)

# algorithm = "right"
# train_accuracies_right_sahand = []
# test_accuracies_right_sahand = []
# for seed = 1:100
#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=seed, chi_init=4, opts=opts, algorithm = algorithm);
#     index = find_stable_accuracy(info["train_acc"], 3)
#     push!(train_accuracies_right_sahand, info["train_acc"][index])
#     push!(test_accuracies_right_sahand, info["test_acc"][index])
# end

# algorithm = "both_one"
# train_accuracies_both_one_sahand = []
# test_accuracies_both_one_sahand = []
# for seed = 1:100
#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=seed, chi_init=4, opts=opts, algorithm = algorithm);
#     index = find_stable_accuracy(info["train_acc"], 3)
#     push!(train_accuracies_both_one_sahand, info["train_acc"][index])
#     push!(test_accuracies_both_one_sahand, info["test_acc"][index])
# end

# algorithm = "both_two"
# train_accuracies_both_two_sahand = []
# test_accuracies_both_two_sahand = []
# for seed = 1:100
#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=seed, chi_init=4, opts=opts, algorithm = algorithm);
#     index = find_stable_accuracy(info["train_acc"], 3)
#     push!(train_accuracies_both_two_sahand, info["train_acc"][index])
#     push!(test_accuracies_both_two_sahand, info["test_acc"][index])
# end
# left_only = false
# encoding = "Stoudenmire"
# opts=Options(; nsweeps=nsweeps, chi_max=chi_max,  update_iters=1, verbosity=verbosity, dtype=Rdtype, lg_iter=KLD_iter,
# bbopt=BBOpt("CustomGD"), track_cost=false, eta=eta, rescale = [false, true], d=2, encoding=Encoding(encoding))
# train_accuracies_right_stoud = []
# test_accuracies_right_stoud = []
# for seed = 1:50
#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=seed, chi_init=4, opts=opts, left_only = left_only);
#     index = find_stable_accuracy(info["train_acc"], 2.5)
#     push!(train_accuracies_right_stoud, info["train_acc"][index])
#     push!(test_accuracies_right_stoud, info["test_acc"][index])
# end

# left_only = true
# encoding = "Sahand"
# opts=Options(; nsweeps=nsweeps, chi_max=chi_max,  update_iters=1, verbosity=verbosity, dtype=Complex{Rdtype}, lg_iter=KLD_iter,
# bbopt=BBOpt("CustomGD"), track_cost=false, eta=eta, rescale = [false, true], d=2, encoding=Encoding(encoding))
# train_accuracies_left_sahand = []
# test_accuracies_left_sahand = []
# for seed = 1:50
#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=seed, chi_init=4, opts=opts, left_only = left_only);
#     index = find_stable_accuracy(info["train_acc"], 2.5)
#     push!(train_accuracies_left_sahand, info["train_acc"][index])
#     push!(test_accuracies_left_sahand, info["test_acc"][index])
# end

# left_only = false
# encoding = "Sahand"
# opts=Options(; nsweeps=nsweeps, chi_max=chi_max,  update_iters=1, verbosity=verbosity, dtype=Complex{Rdtype}, lg_iter=KLD_iter,
# bbopt=BBOpt("CustomGD"), track_cost=false, eta=eta, rescale = [false, true], d=2, encoding=Encoding(encoding))
# train_accuracies_right_sahand = []
# test_accuracies_right_sahand = []
# for seed = 1:50
#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, X_test, y_test; random_state=seed, chi_init=4, opts=opts, left_only = left_only);
#     index = find_stable_accuracy(info["train_acc"], 2.5)
#     push!(train_accuracies_right_sahand, info["train_acc"][index])
#     push!(test_accuracies_right_sahand, info["test_acc"][index])
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
# title!("Bird/Chicken - PBC Going $title - \$\\chi_{max}=$chi_max\$ - \$\\eta=$eta\$ \n $encoding Encoding With Random MPS Initialisation", titlefontsize=13)
# savefig("bird_chicken_chi=$(chi_max)_$(nsweeps)sweeps_$(savetitle)_$(encoding)_random_MPS_initialisation.pdf")
# savefig("bird_chicken_chi=$(chi_max)_$(nsweeps)sweeps_$(savetitle)_$(encoding)_random_MPS_initialisation.png")

# info["test_acc"] = info["test_acc"][1:end-1]
# info["train_acc"] = info["train_acc"][1:end-1]
# x = range(0, stop = nsweeps, length = nsweeps+1)
# plot(x, info["test_acc"], label = "Testing Accuracy", legend=:best)
# plot!(x, info["train_acc"], label = "Training Accuracy")
# xlabel!("Sweeps")
# ylabel!("Accuracy")
# title!("Bird/Chicken - PBC Going $title - \$\\chi_{max}=$chi_max\$ \n \$\\eta=$eta\$ - $encoding Encoding")
# savefig("bird_chicken_chi=$(chi_max)_$(nsweeps)sweeps_$(savetitle)_$(encoding)_random_MPS_initialisation.pdf")
# savefig("bird_chicken_chi=$(chi_max)_$(nsweeps)sweeps_$(savetitle)_$(encoding)_random_MPS_initialisation.png")
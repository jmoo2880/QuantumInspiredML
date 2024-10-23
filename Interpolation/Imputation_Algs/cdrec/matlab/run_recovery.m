matrix = dlmread("data_miss.txt");
recovered = centroid_recovery(matrix, 1);
reference = dlmread("data_full.txt");

disp("Recovery error:")
disp(norm(recovered - reference))

dlmwrite("data_recovered.txt", recovered, 'precision', '%.6f', 'delimiter', '\t')
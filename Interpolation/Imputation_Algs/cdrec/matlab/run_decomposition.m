matrix = dlmread("matrix_100K.txt");
[L, R, Z] = centroid_decomposition(matrix);
sep=" ";
dlmwrite("matrix_100K.L.txt", L, sep)
dlmwrite("matrix_100K.R.txt", R, sep)
%dlmwrite("matrix_100K.Z.txt", Z, sep)
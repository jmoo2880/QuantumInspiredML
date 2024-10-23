all:
	clang++ -O3 -o cmake-build-debug/cdec -Wall -Werror -Wextra -pedantic -Wconversion -Wsign-conversion -Wimplicit-fallthrough -std=c++14 main.cpp Testing.cpp Performance/Benchmark.cpp MathIO/MatrixReadWrite.cpp Algebra/Matrix.cpp Algebra/Vector.cpp Algebra/CentroidDecomposition.cpp Algebra/SVDecomposition.cpp Algebra/MissingValueRecovery.cpp Stats/Correlation.cpp

parallel:
	clang++ -O3 -o cmake-build-debug/cdec -D multi -fopenmp -Wall -Werror -Wextra -pedantic -Wconversion -Wsign-conversion -Wimplicit-fallthrough -std=c++14 main.cpp Testing.cpp Performance/Benchmark.cpp MathIO/MatrixReadWrite.cpp Algebra/Matrix.cpp Algebra/Vector.cpp Algebra/CentroidDecomposition.cpp Algebra/SVDecomposition.cpp Algebra/MissingValueRecovery.cpp Stats/Correlation.cpp

library:
	clang++ -O3 -fPIC -rdynamic -shared -o cmake-build-debug/libIncCD.so -Wall -Werror -Wextra -pedantic -Wconversion -Wsign-conversion -msse4.2 -std=c++14 Algebra/Matrix.cpp Algebra/Vector.cpp Algebra/CentroidDecomposition.cpp Algebra/SVDecomposition.cpp Algebra/MissingValueRecovery.cpp Stats/Correlation.cpp shared/SharedLibFunctions.cpp

library-parallel:
	clang++ -O3 -fPIC -rdynamic -shared -D multi -fopenmp -o cmake-build-debug/libIncCD.so -Wall -Werror -Wextra -pedantic -Wconversion -Wsign-conversion -msse4.2 -std=c++14 Algebra/Matrix.cpp Algebra/Vector.cpp Algebra/CentroidDecomposition.cpp Algebra/SVDecomposition.cpp Algebra/MissingValueRecovery.cpp Stats/Correlation.cpp shared/SharedLibFunctions.cpp

library-monetdb:
	clang++ -O3 -fPIC -rdynamic -shared -o cmake-build-debug/libIncCDMdb.so -Wall -Werror -Wextra -pedantic -Wconversion -Wsign-conversion -msse4.2 -std=c++14 Algebra/Matrix.cpp Algebra/Vector.cpp Algebra/CentroidDecomposition.cpp Algebra/SVDecomposition.cpp Algebra/MissingValueRecovery.cpp Stats/Correlation.cpp shared/SharedLibFunctions.cpp shared/MonetDB.cpp

clean:
	rm cmake-build-debug/incCD cmake-build-debug/libIncCD.so

cmake_minimum_required(VERSION 3.9)
project(incCD)

set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_CXX_STANDARD 17)

set (CMAKE_CXX_FLAGS "-D multi_ -Wall -Werror -Wextra -pedantic -Wconversion -Wsign-conversion -Wimplicit-fallthrough -O3 -msse4.2 -fopenmp")

add_executable(
        incCD

        main.cpp
        MathIO/CommandLine.hpp
        Testing.cpp Testing.h

        Performance/Benchmark.cpp Performance/Benchmark.h
        MathIO/MatrixReadWrite.cpp MathIO/MatrixReadWrite.h

        Algebra/Matrix.cpp Algebra/Matrix.h
        Algebra/Vector.cpp Algebra/Vector.h

        Algebra/CentroidDecomposition.cpp Algebra/CentroidDecomposition.h
        Algebra/SVDecomposition.cpp Algebra/SVDecomposition.h

        Algebra/MissingBlock.hpp
        Algebra/MissingValueRecovery.cpp Algebra/MissingValueRecovery.h

        Stats/Correlation.cpp Stats/Correlation.h
        )

add_library(
        IncCD SHARED

        Algebra/Matrix.cpp Algebra/Matrix.h
        Algebra/Vector.cpp Algebra/Vector.h

        Algebra/CentroidDecomposition.cpp Algebra/CentroidDecomposition.h
        Algebra/SVDecomposition.cpp Algebra/SVDecomposition.h

        Algebra/MissingBlock.hpp
        Algebra/MissingValueRecovery.cpp Algebra/MissingValueRecovery.h

        Stats/Correlation.cpp Stats/Correlation.h
        shared/SharedLibFunctions.cpp shared/SharedLibFunctions.h
        )

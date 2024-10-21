#include <iostream>
#include <tuple>
#include <chrono>

#include "Testing.h"
#include "MathIO/CommandLine.hpp"
#include "MathIO/MatrixReadWrite.h"
#include "Performance/Benchmark.h"

using namespace std;

/* TODO[]
 * - submatrices
 * - persistent normalization for inc. recovery (lockable manually)
 * - add dimensional checks for all mat-mat, vec-vec, mat-vec operations, replace assert with a custom exception
 * - move outer product into vector.h/cpp
 * // todo add element-wise ops
 */

int main(int argc, char *argv[])
{
    // test suite
    
    if (argc == 1)
    {
        Testing::TestBasicActions();
        cout << endl << "---=========---" << endl << endl;
        Testing::TestBasicOps();
        cout << endl << "---=========---" << endl << endl;
        Testing::TestCD();
        cout << endl << "---=========---" << endl << endl;
        Testing::TestIncCD();
        cout << endl << "---=========---" << endl << endl;
        Testing::TestCorr();
        cout << endl << "---=========---" << endl << endl;
        Testing::TestCD_RMV();
        return EXIT_SUCCESS;
    }
    
    // CLI parsing
    
    PAction action = PAction::Undefined;
    PTestType test = PTestType::Undefined;
    PTestParam testParams = PTestParam();
    std::string input;
    std::string output;
    
    uint64_t n = 0, m = 0, k = 0;
    
    uint64_t inc = 0, max = 0;
    uint64_t rcdopt = 0;
    string signVector;
    
    int cliret = CommandLine2(
            argc, argv,
            action, test,
            testParams,
            input, output,
            n, m, k,
            inc, max,
            rcdopt, signVector
    );
    
    #if false
    cout << "Service dump [code=" << cliret << "]" << endl
         << "n = " << n << " m = " << m << " k = " << k << endl
         << "in = " << input << " out = " << output << endl
         << "act = " << (int)action << " test = " << (int)test << endl
         << endl;
    #endif
    
    if (cliret != EXIT_SUCCESS) // parsing failure, command misuse or help request
    {
        return cliret;
    }
    
    // Trivial information
    if (action == PAction::Undefined)
    {
        std::cout << "Action not specified" << std::endl;
        printUsage();
        return EXIT_FAILURE;
    }
    
    if (test == PTestType::Undefined)
    {
        std::cout << "Test type not specified" << std::endl;
        printUsage();
        return EXIT_FAILURE;
    }
    
    if (input.empty())
    {
        std::cout << "Input file is not specified" << std::endl;
        printUsage();
        return EXIT_FAILURE;
    }

    if (output.empty())
    {
        if (test == PTestType::Runtime)
        {
            output = input + "_time.txt";
        }
        else if (test == PTestType::Precision)
        {
            output = input + "_prec.txt";
        }
        else
        {
            switch (action)
            {
                case PAction::Decomposition:
                    output = input;
                    break;
                
                case PAction::Recovery:
                    output = input + "_recov.txt";
                    break;

                case PAction::Normalization:
                    output = input + "_normal.txt";
                    break;
                
                default:
                    break;
            }
        }
    }
    
    // by this point we have an absolute minimum to proceed
    // now we test for specific incompatibilities that don't require matrix size knowledge
    
    if (testParams.Incremental)
    {
        if (n == 0)
        {
            std::cout << "For incremental tests initial rows have to be set manually" << std::endl;
            return EXIT_FAILURE;
        }
        
        if (max > 0 && max <= n)
        {
            std::cout << "For incremental tests incremental maximum has to be 0 or larger than n" << std::endl;
            return EXIT_FAILURE;
        }
    }
    
    // now we load the matrix (fixed or not) and determine the remaining parameters
    
    MathIO::MatrixReader reader(input, '\0'); // zero char means it'll try to auto-detect
    
    if (!reader.isValid())
    {
        return EXIT_FAILURE;
    }
    
    Algebra::Matrix matrix = Algebra::Matrix::empty();
    
    if (n > 0 && m > 0)
    {
        matrix = reader.getFixedMatrix(n, m);
    }
    else if (n > 0)
    {
        matrix = reader.getFixedRowMatrix(n);
        m = matrix.dimM();
    }
    else if (m > 0)
    {
        matrix = reader.getFixedColumnMatrix(m);
        n = matrix.dimN();
    }
    else
    {
        matrix = reader.getFullMatrix();
        n = matrix.dimN();
        m = matrix.dimM();
    }
    
    // parameters that depend on n, m
    
    if (k > m)
    {
        std::cout << "Truncation factor k can't be larger than m" << std::endl;
        return EXIT_FAILURE;
    }
    
    // set defaults because now we can determine those that depend on n, m
    
    if (k == 0)
    {
        k = m;
    }
    
    if (inc == 0)
    {
        inc = 1;
    }
    
    if (max == 0)
    {
        max = testParams.Incremental
              ? n + inc
              : n;
    }
    
    switch (action)
    {
        case PAction::Decomposition:
        {
            Performance::ResultActionDecomposition decomp_res;
            
            if (test == PTestType::Precision)
            {
                decomp_res = Performance::Decomposition(reader, matrix, k, inc, max, signVector);
                
                if (testParams.Incremental || testParams.UseFullOutput)
                {
                    if (testParams.IncrementalLastOnly)
                    {
                        uint64_t lastIdx = decomp_res.Precision.size() - 1;
                        if (testParams.UseFullOutput)
                        {
                            MathIO::exportAnyPrecision(output, n + inc * lastIdx, m, decomp_res.Precision[lastIdx]);
                        }
                        else
                        {
                            MathIO::exportSingleValue(output, decomp_res.Precision[lastIdx]);
                        }
                    }
                    else
                    {
                        for (uint64_t i = 0; i < decomp_res.Precision.size(); ++i)
                        {
                            MathIO::exportAnyPrecision(output, n + inc * i, m, decomp_res.Precision[i]);
                        }
                    }
                }
                else
                {
                    MathIO::exportSingleValue(output, decomp_res.Precision[0]);
                }
            }
            else if (test == PTestType::Runtime)
            {
                decomp_res = Performance::Decomposition(reader, matrix, k, inc, max, signVector);
                
                if (testParams.Incremental || testParams.UseFullOutput)
                {
                    if (testParams.IncrementalLastOnly)
                    {
                        uint64_t lastIdx = decomp_res.Precision.size() - 1;
                        if (testParams.UseFullOutput)
                        {
                            MathIO::exportAnyRuntime(output, n + inc * lastIdx, m, decomp_res.Runtime[lastIdx]);
                        }
                        else
                        {
                            MathIO::exportSingleValue(output, decomp_res.Runtime[lastIdx]);
                        }
                    }
                    else
                    {
                        for (uint64_t i = 0; i < decomp_res.Runtime.size(); ++i)
                        {
                            MathIO::exportAnyRuntime(output, n + inc * i, m, decomp_res.Runtime[i]);
                        }
                    }
                }
                else
                {
                    MathIO::exportSingleValue(output, decomp_res.Runtime[0]);
                }
            }
            else if (test == PTestType::Output)
            {
                decomp_res = Performance::Decomposition(reader, matrix, k, inc, max, signVector);
                
                MathIO::exportDecompOutput(output, decomp_res.Load, decomp_res.Rel, decomp_res.CentroidValues);
            }
            else
            {
                std::cout << "Incompatible combination of action and test type";
                return EXIT_FAILURE;
            }
            break;
        }
        
        case PAction::Recovery:
        {
            std::vector<int64_t> recov_res;
            
            if (test == PTestType::Runtime)
            {
                recov_res = Performance::Recovery(reader, matrix, k, inc, max, signVector,
                                                  testParams.UseBatchCD, testParams.RecoverNormalized);
                
                if (testParams.Incremental || testParams.UseFullOutput)
                {
                    if (testParams.IncrementalLastOnly)
                    {
                        uint64_t lastIdx = recov_res.size() - 1;
                        if (testParams.UseFullOutput)
                        {
                            MathIO::exportAnyRuntime(output, n + inc * lastIdx, m, recov_res[lastIdx]);
                        }
                        else
                        {
                            MathIO::exportSingleValue(output, recov_res[lastIdx]);
                        }
                    }
                    else
                    {
                        for (uint64_t i = 0; i < recov_res.size(); ++i)
                        {
                            MathIO::exportAnyRuntime(output, n + inc * i, m, recov_res[i]);
                        }
                    }
                }
                else
                {
                    MathIO::exportSingleValue(output, recov_res[0]);
                }
            }
            else if (test == PTestType::Output)
            {
                recov_res = Performance::Recovery(reader, matrix, k, inc, max, signVector,
                                                  testParams.UseBatchCD, testParams.RecoverNormalized);
                
                MathIO::exportMatrix(output, matrix);
            }
            else
            {
                std::cout << "Incompatible combination of action and test type";
                return EXIT_FAILURE;
            }
            break;
        }
        
        case PAction::Normalization:
        {
            int64_t normalization_result;
            
            if (test == PTestType::Runtime)
            {
                normalization_result = Performance::Normalization(matrix);
                
                if (testParams.UseFullOutput)
                {
                    MathIO::exportAnyRuntime(output, n, m, normalization_result);
                }
                else
                {
                    MathIO::exportSingleValue(output, normalization_result);
                }
            }
            else if (test == PTestType::Output)
            {
                (void) Performance::Normalization(matrix);
                MathIO::exportMatrix(output, matrix);
            }
            else
            {
                std::cout << "Incompatible combination of action and test type";
                return EXIT_FAILURE;
            }
            break;
        }
        
        default:
            return EXIT_FAILURE;
    }
    
    std::cout << endl;
    return EXIT_SUCCESS;
}

//
// Created by Zakhar on 16/03/2017.
//

#include <chrono>
#include <iostream>
#include <tuple>

#include "Benchmark.h"
#include "../Algebra/CentroidDecomposition.h"
#include "../Algebra/MissingValueRecovery.h"
#include "../Stats/Correlation.h"

using namespace Algebra;
using namespace Algorithms;

namespace Performance
{

std::vector<int64_t> *benchmarkCDtime(Matrix *mx, uint64_t istep)
{
    CentroidDecomposition cd(*mx);
    std::vector<int64_t> *results = new std::vector<int64_t>();
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    cd.performDecomposition();
    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    
    //batchCD, original matrix
    results->push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    
    for (; istep > 0; --istep)
    {
        std::vector<double> newData = MathIO::getNextValue(mx->dimM());
        cd.increment(newData);
    }
    
    {
        begin = std::chrono::steady_clock::now();
        
        //cd.updateCD();
        
        end = std::chrono::steady_clock::now();
        
        //updateCD, original matrix -> augmented matrix
        results->push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
        
        // CACHED CD
        
        begin = std::chrono::steady_clock::now();
        
        cd.performDecomposition();
        
        end = std::chrono::steady_clock::now();
        
        //cachedCD, augmented matrix
        results->push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
        
        // BATCH CD
        
        Matrix mx2 = mx->copy(); // copy of augmented matrix
        CentroidDecomposition cd2(mx2);
        
        begin = std::chrono::steady_clock::now();
        
        cd2.performDecomposition();
        
        end = std::chrono::steady_clock::now();
        
        //batchCD, augmented matrix
        results->push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    }
    
    return results;
}

////////////////////////////////////////////////////////////

Algorithms::CDSignVectorStrategy getStrategyFromStr(const std::string &signVector)
{
    if (signVector == "issv")
    {
        return CDSignVectorStrategy::ISSVBase;
    }
    else if (signVector == "issv+")
    {
        return CDSignVectorStrategy::ISSVPlusBase;
    }
    else if (signVector == "issv-init")
    {
        return CDSignVectorStrategy::ISSVInit;
    }
    else if (signVector == "issv+-init")
    {
        return CDSignVectorStrategy::ISSVPlusInit;
    }
    else if (signVector == "lsv-base" || signVector == "lsv")
    {
        return CDSignVectorStrategy::LSVBase;
    }
    else if (signVector == "lsv-noinit")
    {
        return CDSignVectorStrategy::LSVNoInit;
    }
    else
    {
        return (Algorithms::CDSignVectorStrategy) 0;
    }
}

ResultActionDecomposition Decomposition(
        MathIO::MatrixReader &reader,
        Algebra::Matrix &mat, uint64_t truncation,
        uint64_t istep, uint64_t max, std::string &signVector)
{
    // Local
    ResultActionDecomposition result = ResultActionDecomposition();
    result.CentroidValues.reserve(truncation);
    CentroidDecomposition cd(mat);
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    
    if (!signVector.empty())
    {
        Algorithms::CDSignVectorStrategy strategy = getStrategyFromStr(signVector);
        
        if (isValidStrategy(strategy))
        {
            cd.strategy = strategy;
        }
        else
        {
            std::cout << "[WARNING] Incorrect sign vector strategy code supplied [code=" << signVector << "]. Falling back to a default choice. See --help for more information." << std::endl;
        }
    }
    
    // BatchCD
    begin = std::chrono::steady_clock::now();
    cd.performDecomposition(&result.CentroidValues);
    end = std::chrono::steady_clock::now();
    
    result.Runtime.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    std::cout << "Time: " << result.Runtime.back() << std::endl;
    
    Matrix reconst = matrix_mult_A_BT(cd.getLoad(), cd.getRel());
    reconst -= mat;
    result.Precision.push_back(reconst.normF());
    
    // Incremental CD
    if (istep != 0 && max != 0)
    {
        std::vector<Vector> newlines = std::vector<Vector>();
        newlines.reserve(istep);
        
        for (uint64_t i = 0; i < istep; ++i)
        {
            newlines.push_back(Vector::empty());
        }
        
        while (mat.dimN() < max)
        {
            // read istep lines
            for (uint64_t i = 0; i < istep; ++i)
            {
                if (!reader.hasNextLine())
                {
                    goto error_reader_exhausted;
                }
                newlines[i] = reader.readNextLine();
            }
            
            for (Vector &v : newlines)
            {
                cd.increment(v);
            }
            
            // BatchCD
            begin = std::chrono::steady_clock::now();
            cd.performDecomposition(&result.CentroidValues);
            end = std::chrono::steady_clock::now();
            
            result.Runtime.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
            
            std::cout << "Time: " << result.Runtime.back() << std::endl;
            
            reconst = matrix_mult_A_BT(cd.getLoad(), cd.getRel());
            reconst -= mat;
            result.Precision.push_back(reconst.normF());
        }
      error_reader_exhausted:;
    }
    
    result.Load = cd.stealLoad();
    result.Rel = cd.stealRel();
    
    return result;
}

std::vector<int64_t> Recovery(
        MathIO::MatrixReader &reader,
        Algebra::Matrix &mat, uint64_t truncation,
        uint64_t istep, uint64_t max,
        std::string &signVector,
        bool useBatch, bool useNormalization)
{
    // Local
    std::vector<int64_t> result;
    MissingValueRecovery rmv(mat);
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    
    if (!signVector.empty())
    {
        Algorithms::CDSignVectorStrategy strategy = getStrategyFromStr(signVector);
        
        if (isValidStrategy(strategy))
        {
            rmv.passSignVectorStrategy(strategy);
        }
        else
        {
            std::cout << "[WARNING] Incorrect sign vector strategy code supplied [code=" << signVector << "]. Falling back to a default choice. See --help for more information." << std::endl;
        }
    }
    
    // First Recovery
    rmv.setReduction(truncation);
    rmv.autoDetectMissingBlocks();
    rmv.disableCaching = useBatch;
    rmv.useNormalization = useNormalization;
    
    begin = std::chrono::steady_clock::now();
    rmv.performRecovery(truncation == mat.dimM());
    end = std::chrono::steady_clock::now();
    
    result.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    std::cout << "Time: " << result.back() << std::endl;
    
    // Incremental RMV
    if (mat.dimN() < max)
    {
        std::vector<Vector> newlines = std::vector<Vector>();
        newlines.reserve(istep);
        
        for (uint64_t i = 0; i < istep; ++i)
        {
            newlines.push_back(Vector::empty());
        }
        
        while (mat.dimN() < max)
        {
            // read istep lines
            for (uint64_t i = 0; i < istep; ++i)
            {
                if (!reader.hasNextLine())
                {
                    goto error_reader_exhausted;
                }
                newlines[i] = reader.readNextLine();
            }
            
            for (Vector &v : newlines)
            {
                rmv.increment(v);
            }
    
            rmv.autoDetectMissingBlocks(); // last ones are recovered and thus purged
            
            // BatchCD
            begin = std::chrono::steady_clock::now();
            rmv.performRecovery(truncation == mat.dimM());
            end = std::chrono::steady_clock::now();
            
            result.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
            std::cout << "Time: " << result.back() << std::endl;
        }
      error_reader_exhausted:;
    }
    
    return result;
}

int64_t Normalization(Algebra::Matrix &mat)
{
    Stats::CorrelationMatrix cm(mat);
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    
    begin = std::chrono::steady_clock::now();
    cm.normalizeMatrix();
    end = std::chrono::steady_clock::now();
    
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
}

} // namespace Performance
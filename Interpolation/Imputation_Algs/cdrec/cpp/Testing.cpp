//
// Created by Zakhar on 07.03.2017.
//

#include <string>
#include <iostream>

#include "Testing.h"
#include "Algebra/Matrix.h"
#include "Algebra/CentroidDecomposition.h"
#include "Algebra/MissingValueRecovery.h"
#include "Stats/Correlation.h"

using namespace Algebra;
using namespace Algorithms;
using namespace Stats;

namespace Testing
{

namespace DataSets
{
std::vector<std::vector<double>> synth_missing = {
        {-12.00, 8.00,   -4.00,  -8.00},
        {0.00,   0.00,   0.00,   0.00},
        {-48.00, 32.00,  -16.00, -32.00},
        {NAN,    64.00,  -32.00, -64.00},
        {NAN,    24.00,  -12.00, -24.00},
        {NAN,    64.00,  -32.00, -64.00},
        {NAN,    16.00,  -8.00,  -16.00},
        {NAN,    8.00,   -4.00,  -8.00},
        {NAN,    -32.00, 16.00,  32.00},
        {NAN,    8.00,   -4.00,  -8.00},
        {12.00,  -8.00,  4.00,   8.00},
        {NAN,    -24.00, 12.00,  24.00},
        {NAN,    16.00,  -8.00,  -16.00},
        {-12.00, 8.00,   -4.00,  -8.00},
        {24.00,  -16.00, 8.00,   16.00},
        {NAN,    -8.00,  4.00,   8.00},
        {NAN,    -12.00, 6.00,   12.00},
        {NAN,    -24.00, 12.00,  24.00},
        {48.00,  -32.00, 16.00,  32.00},
        {12.00,  -8.00,  4.00,   8.00}
};

std::vector<std::vector<double>> synth_complete = {
        {-12.00, 8.00,   -4.00,  -8.00},
        {0.00,   0.00,   0.00,   0.00},
        {-48.00, 32.00,  -16.00, -32.00},
        {-96,    64.00,  -32.00, -64.00},
        {-36,    24.00,  -12.00, -24.00},
        {-96,    64.00,  -32.00, -64.00},
        {-24,    16.00,  -8.00,  -16.00},
        {-12,    8.00,   -4.00,  -8.00},
        {48,     -32.00, 16.00,  32.00},
        {-12,    8.00,   -4.00,  -8.00},
        {12.00,  -8.00,  4.00,   8.00},
        {36,     -24.00, 12.00,  24.00},
        {-24,    16.00,  -8.00,  -16.00},
        {-12.00, 8.00,   -4.00,  -8.00},
        {24.00,  -16.00, 8.00,   16.00},
        {12,     -8.00,  4.00,   8.00},
        {18,     -12.00, 6.00,   12.00},
        {36,     -24.00, 12.00,  24.00},
        {48.00,  -32.00, 16.00,  32.00},
        {12.00,  -8.00,  4.00,   8.00}
};

std::vector<std::vector<double>> synth_missing_alt = {
        {NAN,    8.00,   -4.00,  -8.00},
        {NAN,    0.00,   0.00,   0.00},
        {NAN,    32.00,  -16.00, -32.00},
        {-96,    64.00,  -32.00, -64.00},
        {NAN,    24.00,  -12.00, -24.00},
        {-96,   NAN,     -32.00, -64.00},
        {-24,   NAN,     -8.00,  -16.00},
        {-12,    8.00,   -4.00,  -8.00},
        {48,     -32.00, 16.00,  32.00},
        {-12,   NAN,     -4.00,  -8.00},
        {12.00, NAN,     4.00,   8.00},
        {36,    NAN, NAN,       NAN},
        {-24,    16.00,  -8.00, NAN},
        {-12.00, 8.00,   -4.00, NAN},
        {24.00,  -16.00, 8.00,  NAN},
        {12,     -8.00,  4.00,   8.00},
        {18,     -12.00, 6.00,   12.00},
        {36,     -24.00, 12.00,  24.00},
        {48.00, NAN, NAN,        32.00},
        {12.00,  -8.00,  4.00,   8.00}
};
}

//
// Test scenarios
//

void TestCorr()
{
    Matrix mx(DataSets::synth_complete);
    Stats::CorrelationMatrix correlationMatrix(mx);
    
    Matrix &res = correlationMatrix.getCorrelationMatrix();
    std::cout << "Corr(X) =" << std::endl << res.toString() << std::endl;
    
    Vector sv = correlationMatrix.getSingularValuesOfCM();
    std::cout << "Sigma(Corr(X)) =" << std::endl << sv.toString() << std::endl;
}

void TestCD_RMV()
{
    Matrix mx(DataSets::synth_missing_alt);
    
    std::cout << "X =" << std::endl << mx.toString() << std::endl;
    
    MissingValueRecovery mvr(mx, 100, 0.001); //34, 45
    
    mvr.autoDetectMissingBlocks();
    
    //mvr.addMissingBlock(0, 3, 7);  // MB_reference = (-96.00; -36.00; -96.00; -24.00; -12.00; 48.00; -12.00;)^T
    //mvr.addMissingBlock(0, 11, 2); // MB_reference = (36.00; -24.00)^T
    //mvr.addMissingBlock(0, 15, 3); // MB_reference = (12.00; 18.00; 36.00)^T
    
    mvr.setReduction(1);
    
    mvr.performRecovery(true);
    
    std::cout << "X_rec =" << std::endl << mx.toString() << std::endl;
}

namespace DataSets
{
std::vector<std::vector<double>> example1 = {
        {-5.63, -1.58, -6.57},
        {-3.37, -0.20, -3.92},
        {-0.82, 4.07,  1.45}
    
};

std::vector<std::vector<double>> example1full = {
        {-5.63, -1.58, -6.57},
        {-3.37, -0.20, -3.92},
        {-0.82, 4.07,  1.45},
        // example1 ^
        {-1.00, -0.02, -4.77},
        {-2.28, -1.60, -3.63},
        {-2.35, 0.20,  -3.18},
        {-1.37, 4.47,  1.68},
        {-5.05, 0.48,  -5.27},
        {-5.43, -1.23, -7.45},
        {-4.82, 0.52,  -4.90},
        {-4.20, 4.77,  -0.55},
        {-5.07, 0.75,  -5.88}
};

std::vector<std::vector<double>> example1full_ext = {
        {-5.63, -1.58, -6.57, 2.98},
        {-3.37, -0.20, -3.92, -4.01},
        {-0.82, 4.07,  1.45,  3.11},
        // example1 ^
        {-1.00, -0.02, -4.77, 1.11},
        {-2.28, -1.60, -3.63, -2.22},
        {-2.35, 0.20,  -3.18, -1.01},
        {-1.37, 4.47,  1.68,  -0.10},
        {-5.05, 0.48,  -5.27, 0.86},
        {-5.43, -1.23, -7.45, 1.08},
        {-4.82, 0.52,  -4.90, 2.45},
        {-4.20, 4.77,  -0.55, -0.51},
        {-5.07, 0.75,  -5.88, -1.41}
};
}

void TestIncCD()
{
    Matrix mx = Matrix(DataSets::example1); // the same one changed by inc. algo
    
    std::cout << "X =" << std::endl << mx.toString() << std::endl;
    
    CentroidDecomposition cd(mx);
    
    cd.performDecomposition();
    
    const Matrix &L = cd.getLoad();
    const Matrix &R = cd.getRel();
    
    std::cout << "LOAD =" << std::endl << L.toString() << std::endl;
    
    std::cout << "REL =" << std::endl << R.toString() << std::endl;
    
    Matrix reconstructedX = matrix_mult_A_BT(L, R);
    
    reconstructedX -= mx;
    
    std::cout << "||X - (L * R^T)||_F = " << reconstructedX.normF() << std::endl;
    
    for (uint64_t data3iter = 3; data3iter < DataSets::example1full.size(); ++data3iter)
    {
        cd.increment(DataSets::example1full[data3iter]);
        
        cd.performDecomposition();
        
        const Matrix &L2 = cd.getLoad();
        const Matrix &R2 = cd.getRel();
        
        reconstructedX = matrix_mult_A_BT(L2, R2);
        
        reconstructedX -= mx;
        
        std::cout << "Incremental iteration #" << data3iter << std::endl;
        std::cout << "||X - (L * R^T)||_F = " << reconstructedX.normF() << std::endl;
    }
    
    std::cout << "INC:" << std::endl;
    std::cout << cd.getLoad().toString() << std::endl;
    std::cout << cd.getRel().toString() << std::endl;
    
    Matrix mx3 = Matrix(DataSets::example1full);
    CentroidDecomposition cd2(mx3);
    
    cd2.resetSignVectors();
    cd2.performDecomposition();
    
    std::cout << "Batch:" << std::endl;
    std::cout << cd2.getLoad().toString() << std::endl;
    std::cout << cd2.getRel().toString() << std::endl;
    
    Matrix diff = (cd2.getLoad() - cd.getLoad());
    std::cout << "||L - L_i||_F = " << diff.normF() << std::endl;
    
    diff = (cd2.getRel() - cd.getRel());
    std::cout << "||R - R_i||_F = " << diff.normF() << std::endl;
}

void TestCD()
{
    Matrix mx = Matrix(DataSets::example1);
    Matrix mx2 = mx.copy();
    
    std::cout << "X =" << std::endl << mx.toString() << std::endl;
    
    CentroidDecomposition cd(mx2);
    
    cd.performDecomposition();
    
    const Matrix &L = cd.getLoad();
    const Matrix &R = cd.getRel();
    
    std::cout << "LOAD =" << std::endl << L.toString() << std::endl;
    std::cout << "REL =" << std::endl << R.toString() << std::endl;
    
    Matrix reconstructedX = matrix_mult_A_BT(L, R);
    std::cout << "L * R^T: " << std::endl << reconstructedX.toString() << std::endl;
    
    reconstructedX -= mx;
    std::cout << "||X - (L * R^T)||_F = " << reconstructedX.normF() << std::endl;
}

namespace DataSets
{
std::vector<std::vector<double>> testdata1 = {
        {2,  -1, 7},
        {8,  6,  -4},
        {-3, -2, 1}
    
};

std::vector<std::vector<double>> testdata2 = {
        {2,  -2},
        {0,  3},
        {-4, 2}
};

std::vector<double> vector1 = {1.4, -2.0, 0.7};
}

void TestBasicOps()
{
    Matrix m1 = Matrix(DataSets::testdata1);
    Matrix m2 = Matrix(DataSets::testdata2);
    
    Vector v1 = Vector(DataSets::vector1);
    
    Matrix m_id = Matrix::identity(m2.dimN());
    
    std::cout << "M1:" << std::endl << m1.toString() << std::endl;
    std::cout << "M2:" << std::endl << m2.toString() << std::endl;
    std::cout << "V1: " << std::endl << v1.toString() << std::endl;
    std::cout << "ID: " << std::endl << m_id.toString() << std::endl;
    
    // Test 1 - M2 * M1
    
    Matrix resm = m1 * m2;
    
    std::cout << "M1 * M2: " << std::endl << resm.toString() << std::endl;
    
    // Test 2 - M2 * ID
    
    resm = m1 * m_id;
    
    std::cout << "M1 * ID: " << std::endl << resm.toString() << std::endl;
    
    // Test 3 - M1 * V1
    
    std::cout << "V1: " << std::endl << v1.toString() << std::endl;
    
    Vector resv = m1 * v1;
    
    std::cout << "M1 * V1: " << std::endl << resv.toString() << std::endl;
    
    // Test 3 - M2 * V1
    
    resv = m2 ^ v1;
    
    std::cout << "M2^T * V1: " << std::endl << resv.toString() << std::endl;
    
    // Test 4 - scalar & norm of v1
    
    std::cout << "<V1,V1> & ||V1||_2: " << std::endl << vector_dot(v1, v1)
              << " \t" << v1.norm2() << std::endl << std::endl;
    
    // Test 5.1 - normalize and print <,> & norm
    
    v1.normalize();
    
    std::cout << "V1 [normalized]: " << std::endl << v1.toString() << std::endl;
    
    std::cout << "<V1,V1> & ||V1||_2: " << std::endl << vector_dot(v1, v1)
              << " \t" << v1.norm2() << std::endl << std::endl;
}

void TestBasicActions()
{
    Matrix m1 = Matrix(DataSets::testdata1);
    Matrix m2 = Matrix(DataSets::testdata2);
    
    Vector v1 = Vector(DataSets::vector1);
    
    Matrix m_id = Matrix::identity(m2.dimN());
    
    std::cout << "M1:" << std::endl << m1.toString() << std::endl;
    std::cout << "M2:" << std::endl << m2.toString() << std::endl;
    std::cout << "V1: " << std::endl << v1.toString() << std::endl;
    std::cout << "ID: " << std::endl << m_id.toString() << std::endl;
    
    
    Vector diag = m_id.diag().diag().diag().diag().diag().diag().diag();
    std::cout << "diag(diag(diag(M1))): " << std::endl << diag.toString() << std::endl;
    
    Matrix id_cut = m_id.subMatrix(2, 2);
    std::cout << "ID(:2, :2): " << std::endl << id_cut.toString() << std::endl;
    
    Matrix m1_cut_fullcol = m1.subMatrix(2, 3);
    std::cout << "M1(:2, :): " << std::endl << m1_cut_fullcol.toString() << std::endl;
    
    Matrix m1_cut = m1.subMatrix(1, 1, 2, 2);
    std::cout << "M1(2:3, 2:3): " << std::endl << m1_cut.toString() << std::endl;
    
    Matrix m2_cut = m2.subMatrix(1, 1, 2, 1);
    std::cout << "M2(2:3, 2:2): " << std::endl << m2_cut.toString() << std::endl;
    
    Matrix m2_cut_fullcol = m2.subMatrix(1, 0, 1, 2);
    std::cout << "M2(2, :): " << std::endl << m2_cut_fullcol.toString() << std::endl;
    
    Vector v1_cut = v1.subVector(2);
    std::cout << "V1(:2): " << std::endl << v1_cut.toString() << std::endl;
    
    Vector v1_cut_alt = v1.subVector(1, 2);
    std::cout << "V1(2:3): " << std::endl << v1_cut_alt.toString() << std::endl;
}

} //namespace Testing

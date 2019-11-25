#ifndef TEST_H
#define TEST_H

#include "decomposition.h"
class Test
{
public:
    Test();
    static void testGauss();
    static void testLU();
    static void testEij();
    static void testCholesky();
    static void testGivensQR();
    static void testHousholderQR();
    static void testCholeskyInverse();
    static void testQRInverse();
    static void test3x3Inverse();
    static void testPseudoInverse();
    static void testSolveMatrix();
    static void testSolveGenesis2BFM();
    static void testPseudoinverse();
};

#endif // TEST_H

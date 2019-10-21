#ifndef TEST_H
#define TEST_H

#include "decomposition.h"
class Test
{
public:
    Test();
    static void testGauss();
    static void testLU();
    static void testCholesky();
    static void testGivensQR();
    static void testCholeskyInverse();
};

#endif // TEST_H

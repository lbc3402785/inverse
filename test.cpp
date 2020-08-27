#include "test.h"

Test::Test()
{

}

void Test::testGauss()
{
    Eigen::Matrix<float,6,6> C=Eigen::Matrix<float,6,6>::Random(6,6);
    Eigen::Matrix<float,6,6> A=C.transpose()*C+0.1*Eigen::Matrix<float,6,6>::Identity();

    std::cout<<"-------"<<std::endl;
    std::cout<<A.inverse()<<std::endl;
    std::cout<<"-------"<<std::endl;
    std::cout<<Decomposition<float,A.Options>::Gauss(A)<<std::endl;
}

void Test::testLU()
{
    //    Eigen::Matrix<float,3,3> A=Eigen::Matrix<float,3,3>::Zero(3,3);
    //    A<<4,8,4,2,7,2,1,2,3;
    //    std::cout<<"A:"<<std::endl<<A<<std::endl;
    //    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> L;//=Eigen::Matrix<float,3,3>::Zero(3,3);
    //    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> U;//=Eigen::Matrix<float,3,3>::Zero(3,3);
    //    Decomposition<float,A.Options>::LU(A,L,U);
    //    std::cout<<"L:"<<std::endl<<L<<std::endl;
    //    std::cout<<"U:"<<std::endl<<U<<std::endl;
    Eigen::Matrix<float,3,3> A=Eigen::Matrix<float,3,3>::Random(3,3);
        A=A.transpose()*A+0.1*Eigen::Matrix<float,3,3>::Identity();

    std::cout<<"A:"<<std::endl<<A<<std::endl;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> L;//=Eigen::Matrix<float,3,3>::Zero(3,3);
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> U;//=Eigen::Matrix<float,3,3>::Zero(3,3);
    Decomposition<float,A.Options>::LU(A,L,U,LUType::Doolittle);
    std::cout<<"L:"<<std::endl<<L<<std::endl;
    std::cout<<"U:"<<std::endl<<U<<std::endl;
    std::cout<<"L*U:"<<std::endl<<L*U<<std::endl;
}

void Test::testCompactLU()
{
    Eigen::Matrix<float,5,5> A;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> L;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> U;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> P;
//    A<<1,2,3,3,1,5,2,5,2;
     A=Eigen::Matrix<float,5,5>::Random(5,5);
        A=A.transpose()*A+0.1*Eigen::Matrix<float,5,5>::Identity();

    Decomposition<float,A.Options>::compatDoolittleLU(A,L,U,P);
    std::cout<<"L:"<<std::endl<<L<<std::endl;
    std::cout<<"U:"<<std::endl<<U<<std::endl;
    std::cout<<"L*U:"<<std::endl<<L*U<<std::endl;
    std::cout<<"P*A:"<<std::endl<<P*A<<std::endl;
}

void Test::testEij()
{
    Eigen::Matrix<float,5,5> e1=Decomposition<float>::EijMatrix(5,1,3);
    std::cout<<"e1:"<<std::endl<<e1<<std::endl;
    Eigen::Matrix<float,5,5> e2=Decomposition<float>::EijMatrix(5,3,1);
    std::cout<<"e2:"<<std::endl<<e2<<std::endl;
}

void Test::testCholesky()
{
    Eigen::Matrix<float,5,5> A=Eigen::Matrix<float,5,5>::Random(5,5);
    A=A.transpose()*A+0.1*Eigen::Matrix<float,5,5>::Identity();
    std::cout<<"A:"<<std::endl<<A<<std::endl;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> L;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> D;
    Decomposition<float,A.Options>::LDLT(A,L,D);
    std::cout<<"L*D*L^T:"<<std::endl<<L*D*L.transpose()<<std::endl;
}

void Test::testGivensQR()
{
    Eigen::Matrix<float,5,5> A=Eigen::Matrix<float,5,5>::Random(5,5);
    A=A.transpose()*A+0.1*Eigen::Matrix<float,5,5>::Identity();
    std::cout<<"A:"<<std::endl<<A<<std::endl;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> Q;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> R;
    Decomposition<float,A.Options>::GivensQR(A,Q,R);
    std::cout<<"Q:"<<std::endl<<Q<<std::endl;
    std::cout<<"R:"<<std::endl<<R<<std::endl;
    std::cout<<"Q*R:"<<std::endl<<Q*R<<std::endl;
}

void Test::testHousholderQR()
{
    //    Eigen::Matrix<float,3,3> A;
    //    A<<2,2,1,1,2,2,2,1,2;
    Eigen::Matrix<float,5,5> A=Eigen::Matrix<float,5,5>::Random(5,5);
    A=A.transpose()*A+0.1*Eigen::Matrix<float,5,5>::Identity();
    std::cout<<"A:"<<std::endl<<A<<std::endl;

    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> Q;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> R;
    Decomposition<float,A.Options>::HousholderQR(A,Q,R);
    std::cout<<"Q:"<<std::endl<<Q<<std::endl;
    std::cout<<"R:"<<std::endl<<R<<std::endl;
    std::cout<<"Q*R:"<<std::endl<<Q*R<<std::endl;
}

void Test::testCholeskyInverse()
{
    Eigen::Matrix<float,5,5> A=Eigen::Matrix<float,5,5>::Random(5,5);
    A=A.transpose()*A+0.1*Eigen::Matrix<float,5,5>::Identity();
    std::cout<<"A:"<<std::endl<<A<<std::endl;
    std::cout<<"A.inverse():"<<std::endl<<A.inverse()<<std::endl;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> O;
    Decomposition<float,A.Options>::CholeskyInverse(A,O);
    std::cout<<"O1:"<<std::endl<<O<<std::endl;
    Decomposition<float,A.Options>::CholeskyMulInverse(A,O);
    std::cout<<"O2:"<<std::endl<<O<<std::endl;
}

void Test::testQRInverse()
{
    Eigen::Matrix<float,5,5> A=Eigen::Matrix<float,5,5>::Random(5,5);
    A=A.transpose()*A+0.1*Eigen::Matrix<float,5,5>::Identity();
    std::cout<<"A:"<<std::endl<<A<<std::endl;
    std::cout<<"A.inverse():"<<std::endl<<A.inverse()<<std::endl;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> O;
    Decomposition<float,A.Options>::QRInverse(A,O);
    std::cout<<"O:"<<std::endl<<O<<std::endl;
}

void Test::test3x3Inverse()
{
    Eigen::Matrix<float,3,3> A=Eigen::Matrix<float,3,3>::Random(3,3);
    A=A.transpose()*A+0.1*Eigen::Matrix<float,3,3>::Identity();
    std::cout<<"A:"<<std::endl<<A<<std::endl;
    std::cout<<"A.inverse():"<<std::endl<<A.inverse()<<std::endl;
    std::cout<<"A.inverse():"<<std::endl<<Decomposition<float,A.Options>::matrix3x3Inverse(A,Decomposition<float,A.Options>::matrix3x3Determinant(A))<<std::endl;
}

void Test::testPseudoInverse()
{
    Eigen::Vector3f x(1,2,3);
    Eigen::Matrix<float,3,3> A=x*x.transpose();
    std::cout<<"A:"<<std::endl<<A<<std::endl;
    std::cout<<"pseudoInverse:"<<std::endl<<A.completeOrthogonalDecomposition().solve(A*x)<<std::endl;
}

void Test::testSolveMatrix()
{
    Eigen::Vector3f x(1,2,3);
    Eigen::Matrix<float,3,3> A=x*x.transpose();
    Eigen::Matrix<float,3,3> I=Eigen::Matrix<float,3,3>::Identity();
    A+=I;
    std::cout<<"A:"<<std::endl<<A<<std::endl;
    Eigen::Vector3f x0(5,1,4);
    Eigen::Vector3f x1(4,0,2);
    Eigen::Vector3f x2(6,1,9);
    Eigen::Matrix<float,3,3> B;
    B.col(0)=A*x0;B.col(1)=A*x1;B.col(2)=A*x2;
    std::cout<<"B:"<<std::endl<<B<<std::endl;
    std::cout<<"C1:"<<std::endl<<A.colPivHouseholderQr().solve(B)<<std::endl;
    std::cout<<"C2:"<<std::endl<<A.completeOrthogonalDecomposition().solve(B)<<std::endl;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> O;
    Decomposition<float,Eigen::ColMajor>::QRInverse(A,O);
    std::cout<<"C3:"<<std::endl<<A.inverse()*B<<std::endl;
    std::cout<<"C4:"<<std::endl<<O*B<<std::endl;
//    std::cout<<"x0:"<<std::endl<<(A).inverse()*B.col(0)<<std::endl;
//    std::cout<<"x1:"<<std::endl<<(A).inverse()*B.col(1)<<std::endl;
    //    std::cout<<"x2:"<<std::endl<<(A).inverse()*B.col(2)<<std::endl;
}

void Test::testSolveGenesis2BFM()
{
    Eigen::Matrix<float,10,1> x=Eigen::Matrix<float,10,1>::Random();
    std::cout<<"x:"<<std::endl<<x<<std::endl;
    Eigen::Matrix<float,5,1> y=Eigen::Matrix<float,5,1>::Zero();
    y(2,0)=1;
    Eigen::Matrix<float,5,10> A=Eigen::Matrix<float,5,10>::Zero();
}

void Test::testPseudoinverse()
{
    Eigen::Matrix<float,3,3> A=Eigen::Matrix<float,3,3>::Random(3,3);
    Eigen::Matrix<float,3,3> B=A*A.transpose();
    std::cout<<"B:"<<std::endl<<B<<std::endl;
    Eigen::Matrix<float,3,3> C=Decomposition<float,Eigen::ColMajor>::pseudoinverse(B);
    std::cout<<"B*C*B:"<<std::endl<<B*C*B<<std::endl;
}

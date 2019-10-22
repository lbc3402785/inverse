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
    Decomposition<float,A.Options>::LU(A,L,U);
//    std::cout<<"L:"<<std::endl<<L<<std::endl;
//    std::cout<<"U:"<<std::endl<<U<<std::endl;
    std::cout<<"L*U:"<<std::endl<<L*U<<std::endl;
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

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

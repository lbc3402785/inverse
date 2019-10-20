#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
template<typename T,int _Options=Eigen::RowMajor>
class Decomposition
{
public:
    Decomposition();
    static Eigen::Matrix<T, Eigen::Dynamic,Eigen::Dynamic, _Options> Gauss(const Eigen::Matrix<T, Eigen::Dynamic,Eigen::Dynamic, _Options>&A);
};
/**
 *————————————————
 *      版权声明：本文为CSDN博主「weixin_40905871」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
 *      原文链接：https://blog.csdn.net/weixin_40905871/article/details/81909887
 */
template<typename T, int _Options>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> Decomposition<T,_Options>::Gauss(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> &A)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> B=A;
    if(A.rows()!=A.cols()){
        throw std::invalid_argument("not a squared matrix");
    }
    int n=A.rows();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> O(n,n);
    O.setIdentity();

    int i,j,k,l;
    T major,temp;
    for(i=0;i<n-1;i++){
        major=B(i,i);
        k=i;
        for(j=i+1;j<n;j++){
            if(std::fabs(B(j,i)>std::fabs(major))){
                major=B(j,j);
                k=j;
            }
        }
        if (k != i)
        {
            for (j = 0; j < n; j++)
            {
                temp = B(i,j);
                B(i,j) = B(k,j);
                B(k,j) = temp;
                //O伴随交换
                temp = O(i,j);
                O(i,j) = O(k,j);
                O(k,j) = temp;
            }
        }
        //判断主元是否为0, 若是, 则矩阵A不是满秩矩阵,不存在逆矩阵
        if(std::fabs(B(i,i))<1e-8){
            throw std::invalid_argument("not a non-singular matrix");
        }
        temp=B(i,i);
        //消去B的第i列除去i行以外的各行元素
        for(j=0;j<n;j++){
            B(i,j) /= temp;        //主对角线上的元素变为1
            O(i,j) /= temp;        //伴随计算
        }
        for(j=i+1;j<n;j++){
            temp=B(j,i);
            for(k=0;k<n;k++){
                B(j,k)-=temp*B(i,k);
                O(j,k)-=temp*O(i,k);
            }
        }
    }
    temp=B(i,i);
    //消去B的第i列除去i行以外的各行元素
    for(j=0;j<n;j++){
        B(i,j) /= temp;        //主对角线上的元素变为1
        O(i,j) /= temp;        //伴随计算
    }
//    std::cout<<i<<"--"<<std::endl<<B<<std::endl;
    for(i=n-1;i>0;i--){

        for(j=i-1;j>=0;j--){
            temp=B(j,i);
            if(temp==0)continue;
            for(k=0;k<n;k++){
                B(j,k)-=temp*B(i,k);
                O(j,k)-=temp*O(i,k);
            }
        }
    }
//    std::cout<<i<<"--"<<std::endl<<B<<std::endl;
    return O;
}
#endif // DECOMPOSITION_H

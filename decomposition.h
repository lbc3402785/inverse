#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
enum QRType{Givens,HouseHolder};
enum CholeskyType{CholeskyLLT,CholeskyLDLT};
enum LUType{Crout,Doolittle};
template<typename T,int _Options=Eigen::RowMajor>
class Decomposition
{
public:
    Decomposition();
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> MatrixXXT;
    typedef Eigen::Matrix<T, 3, 3, Eigen::RowMajor> Matrix3x3T;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1, _Options> VectorXT;
    static MatrixXXT Gauss(const MatrixXXT&A);
    static MatrixXXT pseudoinverse(const MatrixXXT&A);
    static MatrixXXT EijMatrix(int n,int i,int j);
    static MatrixXXT RijMatrix(int n,int i,int j,T cos,T sin);
    static MatrixXXT RijInverseMatrix(int n,int i,int j,T cos,T sin);
    static MatrixXXT reflectionMatrix(VectorXT& a);
    static void lowerDiagonalInverse(const MatrixXXT&L,MatrixXXT&U);
    static void upperDiagonalInverse(const MatrixXXT&L,MatrixXXT&U);
    static void sqrtDiagonal(const MatrixXXT&L,MatrixXXT&O);
    static void invertDiagonal(const MatrixXXT&L,MatrixXXT&O);
    static bool TridiagonalLU(const MatrixXXT&A,MatrixXXT& L,MatrixXXT&U);
    static bool LU(const MatrixXXT&A,MatrixXXT& L,MatrixXXT&U,LUType type=Crout);
    static void compatDoolittleLU(const MatrixXXT&A,MatrixXXT& L,MatrixXXT&U,MatrixXXT&P);
    static void GivensQR(const MatrixXXT&A,MatrixXXT& Q,MatrixXXT&R);
    static void HousholderQR(const MatrixXXT&A,MatrixXXT& Q,MatrixXXT&R);
    static void LLT(const MatrixXXT&A,MatrixXXT& L);
    static void LDLT(const MatrixXXT&A,MatrixXXT& L,MatrixXXT& D);
    static void CholeskyInverse(const MatrixXXT&A,MatrixXXT& O,CholeskyType type=CholeskyLLT);
    static void QRInverse(const MatrixXXT&A,MatrixXXT& O,QRType type=HouseHolder);
    static void CholeskyMulInverse(const MatrixXXT&A,MatrixXXT& O);
    static void lowerDiagonalTranposeMul(const MatrixXXT& U,MatrixXXT& O);
    static T matrix3x3Determinant(const Matrix3x3T &A);
    static Matrix3x3T matrix3x3Inverse(const Matrix3x3T &A,T det);
};
/**
 *————————————————
 *      版权声明：本文为CSDN博主「weixin_40905871」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
 *      原文链接：https://blog.csdn.net/weixin_40905871/article/details/81909887
 */
template<typename T, int _Options>
typename Decomposition<T,_Options>::MatrixXXT Decomposition<T,_Options>::Gauss(const MatrixXXT &A)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> B=A;
    if(A.rows()!=A.cols()){
        throw std::invalid_argument("not a squared matrix");
    }
    int n=A.rows();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> O(n,n);
    O.setIdentity();

    int i,j,k;
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

template<typename T, int _Options>
typename Decomposition<T,_Options>::MatrixXXT Decomposition<T,_Options>::pseudoinverse(const Decomposition<T,_Options>::MatrixXXT &A)
{

    auto svd = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto &singularValues = svd.singularValues();
    Decomposition<T,_Options>::MatrixXXT singularValuesInv(A.cols(), A.rows());
    singularValuesInv.setZero();
    T  pinvtoler = 1.e-6; // choose your tolerance wisely
    for (unsigned int i = 0; i < singularValues.size(); ++i) {
        if (singularValues(i) > pinvtoler)
            singularValuesInv(i, i) = 1.0f / singularValues(i);
        else
            singularValuesInv(i, i) = 0.f;
    }
    Decomposition<T,_Options>::MatrixXXT pinvmat = svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
    return pinvmat;
}

template<typename T, int _Options>
typename Decomposition<T,_Options>::MatrixXXT Decomposition<T,_Options>::EijMatrix(int n, int i, int j)
{

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> Eij(n,n);
    Eij.setIdentity();
    if(i!=j){
        Eij(i-1,i-1)=0;
        Eij(i-1,j-1)=1;
        Eij(j-1,i-1)=1;
        Eij(j-1,j-1)=0;
    }
    return Eij;
}

template<typename T, int _Options>
typename Decomposition<T,_Options>::MatrixXXT Decomposition<T,_Options>::RijMatrix(int n, int i, int j, T cos, T sin)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> Rij(n,n);
    Rij.setIdentity();
    Rij(i-1,i-1)=cos;
    Rij(i-1,j-1)=sin;
    Rij(j-1,i-1)=-sin;
    Rij(j-1,j-1)=cos;
    return Rij;
}

template<typename T, int _Options>
typename Decomposition<T,_Options>::MatrixXXT Decomposition<T,_Options>::RijInverseMatrix(int n, int i, int j, T cos, T sin)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> RijInverse(n,n);
    RijInverse.setIdentity();
    RijInverse(i-1,i-1)=cos;
    RijInverse(i-1,j-1)=-sin;
    RijInverse(j-1,i-1)=sin;
    RijInverse(j-1,j-1)=cos;
    return RijInverse;
}

template<typename T, int _Options>
typename Decomposition<T,_Options>::MatrixXXT Decomposition<T,_Options>::reflectionMatrix(Decomposition::VectorXT &a)
{
    int n=a.rows();
    VectorXT  e(n);
    e.setZero();
    e(0)=a.norm();
    VectorXT w=a-e;
    w.normalize();
    MatrixXXT I(n,n);
    I.setIdentity();
    MatrixXXT H=I-2*w*w.transpose();
    return H;
}

template<typename T, int _Options>
void Decomposition<T,_Options>::lowerDiagonalInverse(const Decomposition::MatrixXXT &L, Decomposition::MatrixXXT &U)
{
    if(L.rows()!=L.cols()){
        throw std::invalid_argument("not a squared matrix");
    }
    if(U.rows()!=L.rows()||U.cols()!=L.cols()){
        U.resizeLike(L);
    }
    U.setZero();
    int i,j,k;
    int n=L.rows();
    for(i=1;i<=n;i++){
        if(L(i-1,i-1)==0){
            throw std::invalid_argument("not a non-singular matrix");
        }
        for(j=1;j<i;j++){
            T temp=0;
            for(k=j;k<=i-1;k++){
                temp-=L(i-1,k-1)*U(k-1,j-1);
            }
            U(i-1,j-1)=temp/L(i-1,i-1);
        }
        U(i-1,i-1)=1/L(i-1,i-1);
    }
}

template<typename T, int _Options>
void Decomposition<T,_Options>::upperDiagonalInverse(const Decomposition::MatrixXXT &L, Decomposition::MatrixXXT &U)
{
    if(L.rows()!=L.cols()){
        throw std::invalid_argument("not a squared matrix");
    }
    if(U.rows()!=L.rows()||U.cols()!=L.cols()){
        U.resizeLike(L);
    }
    U.setZero();
    int i,j,k;
    int n=L.rows();
    for(i=1;i<=n;i++){
        if(L(i-1,i-1)==0){
            throw std::invalid_argument("not a non-singular matrix");
        }
        U(i-1,i-1)=1/L(i-1,i-1);
        for(j=i+1;j<=n;j++){
            T temp=0;
            for(k=i;k<=j-1;k++){
                temp-=U(i-1,k-1)*L(k-1,j-1);
            }
            U(i-1,j-1)=temp/L(j-1,j-1);
        }

    }
}

template<typename T, int _Options>
void Decomposition<T,_Options>::sqrtDiagonal(const Decomposition::MatrixXXT &L, Decomposition::MatrixXXT &O)
{
    if(L.rows()!=L.cols()){
        throw std::invalid_argument("not a squared matrix");
    }
    if(O.rows()!=L.rows()||O.cols()!=L.cols()){
        O.resizeLike(L);
    }
    O.setZero();
    int n=L.rows();
    int i,j;
    for(i=1;i<=n;i++){
        if(L(i-1,i-1)<0){
            throw std::invalid_argument("not a positive definite matrix");
        }
        O(i-1,i-1)=std::sqrt(L(i-1,i-1));
    }
}

template<typename T, int _Options>
void Decomposition<T,_Options>::invertDiagonal(const Decomposition::MatrixXXT &L, Decomposition::MatrixXXT &O)
{
    if(L.rows()!=L.cols()){
        throw std::invalid_argument("not a squared matrix");
    }
    if(O.rows()!=L.rows()||O.cols()!=L.cols()){
        O.resizeLike(L);
    }
    O.setZero();
    int n=L.rows();
    int i,j;
    for(i=1;i<=n;i++){
        if(L(i-1,i-1)==0){
            throw std::invalid_argument("not a non-singular matrix");
        }
        O(i-1,i-1)=1/L(i-1,i-1);
    }
}



template<typename T, int _Options>
bool Decomposition<T,_Options>::LU(const MatrixXXT &A, MatrixXXT &L, MatrixXXT &U,LUType type)
{
    if(A.rows()!=A.cols()){
        throw std::invalid_argument("not a squared matrix");
    }
    if(L.rows()!=A.rows()||L.cols()!=A.cols()){
        L.resizeLike(A);
    }
    if(U.rows()!=A.rows()||U.cols()!=A.cols()){
        U.resizeLike(A);
    }
    if(type==Crout){
        L.setZero();
        U.setIdentity();
        int i,j,k;
        int n=A.rows();
        for(i=1;i<=n;i++){
            for(j=1;j<=n;j++){
                T temp=0;

                if(j<=i){
                    for(k=1;k<=j-1;k++){
                        temp+=L(i-1,k-1)*U(k-1,j-1);
                    }
                    L(i-1,j-1)=A(i-1,j-1)-temp;
                }else{
                    for(k=1;k<=i;k++){
                        temp+=L(i-1,k-1)*U(k-1,j-1);
                    }
                    if(L(i-1,i-1)==0)return false;
                    U(i-1,j-1)=(A(i-1,j-1)-temp)/L(i-1,i-1);
                }
            }
        }
    }else{
        L.setIdentity();
        U.setZero();
        U.row(0)=A.row(0);
        if(A(0,0)==0)return false;
        L.col(0)=A.col(0)/A(0,0);
        int k,j,r;
        int n=A.rows();
        for(k=1;k<n;k++){
            for(j=k;j<n;j++){
                T temp=0;
                for(r=0;r<k;r++){
                    temp+=L(k,r)*U(r,j);
                }
                U(k,j)=A(k,j)-temp;

                if(j>k){
                    temp=0;
                    for(r=0;r<k;r++){
                        temp+=L(j,r)*U(r,k);
                    }
                    L(j,k)=(A(j,k)-temp)/U(k,k);
                }
            }
        }
    }
    return true;
}

template<typename T, int _Options>
void Decomposition<T,_Options>::compatDoolittleLU(const Decomposition::MatrixXXT &A, Decomposition::MatrixXXT &L, Decomposition::MatrixXXT &U, Decomposition::MatrixXXT &P)
{
    if(A.rows()!=A.cols()){
        throw std::invalid_argument("not a squared matrix");
    }
    if(L.rows()!=A.rows()||L.cols()!=A.cols()){
        L.resizeLike(A);
    }
    if(U.rows()!=A.rows()||U.cols()!=A.cols()){
        U.resizeLike(A);
    }
    if(P.rows()!=A.rows()||P.cols()!=A.cols()){
        P.resizeLike(A);
    }
    MatrixXXT R=A;
    P.setIdentity();
    int n=A.rows();
    for(int k=0;k<n;k++){
        T sik=0;
        std::vector<T> vs;
        int ik=k;
        for(int i=k;i<n;i++){
            T temp=R(i,k);
            for(int r=0;r<k;r++){
                temp-=R(i,r)*R(r,k);
            }
            vs.push_back(temp);
            if(std::fabs(sik)<temp){
                sik=temp;
                ik=i;
            }
        }
        if(ik!=k){
            T t;
            for(int j=0;j<n;j++){
                t=R(k,j);
                R(k,j)=R(ik,j);
                R(ik,j)=t;
            }

            t=vs[0];
            vs[0]=sik;
            vs[ik-k]=t;
            P=EijMatrix(n,k+1,ik+1)*P;
        }
        R(k,k)=sik;
        for(int j=k+1;j<n;j++){
            for(int r=0;r<k;r++){
                R(k,j)-=R(k,r)*R(r,j);
            }
        }
        for(int i=k+1;i<n;i++){
            R(i,k)=vs[i-k]/sik;
        }
    }
    L.setIdentity();
    U.setZero();
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(j<i){
                L(i,j)=R(i,j);
            }else{
                U(i,j)=R(i,j);
            }
        }
    }
}

template<typename T, int _Options>
void Decomposition<T,_Options>::GivensQR(const MatrixXXT &A, MatrixXXT &Q, MatrixXXT &R)
{
    if(A.rows()!=A.cols()){
        throw std::invalid_argument("not a squared matrix");
    }
    int n=A.rows();
    R=A;
    if(R.rows()!=A.rows()||R.cols()!=A.cols()){
        R.resizeLike(A);
    }
    if(Q.rows()!=A.rows()||Q.cols()!=A.cols()){
        Q.resizeLike(A);
    }
    Q.setIdentity();
    int i,j,k;

    for(i=1;i<=n-1;i++){
        if(R(i-1,i-1)!=0){

        }else{
            for(k=i+1;k<=n;k++){
                if(A(k-1,i-1)!=0){
                    break;
                }
            }
            if(k>n){
                throw std::invalid_argument("not a non-singular matrix");
                break;
            }
            //exchange
            MatrixXXT Eij=EijMatrix(n,i,k);
//            MatrixXXT Eji=EijMatrix(n,k,i);
            R=Eij*R;
            Q*=Eij;
        }
        for(j=i+1;j<=n;j++){
            T a_ii=R(i-1,i-1);
            T a_ji=R(j-1,i-1);
            T cos=a_ii/std::sqrt(a_ii*a_ii+a_ji*a_ji);
            T sin=a_ji/std::sqrt(a_ii*a_ii+a_ji*a_ji);
            MatrixXXT Rij=RijMatrix(n,i,j,cos,sin);
            MatrixXXT RijInverse=RijInverseMatrix(n,i,j,cos,sin);
            R=Rij*R;
            Q*=RijInverse;
        }

    }
}

template<typename T, int _Options>
void Decomposition<T,_Options>::HousholderQR(const Decomposition::MatrixXXT &A, Decomposition::MatrixXXT &Q, Decomposition::MatrixXXT &R)
{
    if(A.rows()!=A.cols()){
        throw std::invalid_argument("not a squared matrix");
    }
    int n=A.rows();
    R=A;
    if(Q.rows()!=A.rows()||Q.cols()!=A.cols()){
        Q.resizeLike(A);
    }
    Q.setIdentity();
    MatrixXXT I(n,n);
    I.setIdentity();
    int i,j,k;
    for(i=1;i<=n-1;i++){
        VectorXT a=R.block(i-1,i-1,n-i+1,1);
        MatrixXXT H=I;
        MatrixXXT Hn=reflectionMatrix(a);
        H.block(i-1,i-1,n-i+1,n-i+1)=Hn;
        Q*=H.transpose();
        R=H*R;
        //R.block(i,i-1,n-i,1).setZero();
    }
    for(i=1;i<=n-1;i++){
      R.block(i,i-1,n-i,1).setZero();
    }
}



template<typename T, int _Options>
void Decomposition<T,_Options>::CholeskyInverse(const Decomposition::MatrixXXT &A, Decomposition::MatrixXXT &O,CholeskyType type)
{
    if(type==CholeskyLLT){
        MatrixXXT L;
        LLT(A,L);
        MatrixXXT U;
        lowerDiagonalInverse(L,U);
        lowerDiagonalTranposeMul(U,O);
    }else if(type==CholeskyLDLT){
        MatrixXXT L;
        MatrixXXT D;
        LDLT(A,L,D);
        MatrixXXT Dsqrt;
        sqrtDiagonal(D,Dsqrt);
        L*=Dsqrt;
        MatrixXXT U;
        lowerDiagonalInverse(L,U);
        lowerDiagonalTranposeMul(U,O);
    }

}

template<typename T, int _Options>
void Decomposition<T,_Options>::QRInverse(const Decomposition::MatrixXXT &A, Decomposition::MatrixXXT &O,QRType type)
{
    MatrixXXT Q;
    MatrixXXT R;
    if(type==HouseHolder){
        HousholderQR(A,Q,R);
    }else if(type==Givens){
        GivensQR(A,Q,R);
    }else{
        throw std::invalid_argument("invalid QRType:use HouseHolder or Givens");
    }


    MatrixXXT Rinverse;
    upperDiagonalInverse(R,Rinverse);
    O=Rinverse*Q.transpose();
}

template<typename T, int _Options>
void Decomposition<T,_Options>::CholeskyMulInverse(const Decomposition::MatrixXXT &A, Decomposition::MatrixXXT &O)
{
    MatrixXXT L;
    MatrixXXT D;
    LDLT(A,L,D);
    MatrixXXT Dinvert;
    invertDiagonal(D,Dinvert);
    MatrixXXT U;
    lowerDiagonalInverse(L,U);
    O=U.transpose()*Dinvert*U;
}

template<typename T, int _Options>
void Decomposition<T,_Options>::lowerDiagonalTranposeMul(const Decomposition::MatrixXXT &U, Decomposition::MatrixXXT &O)
{
    if(U.rows()!=U.cols()){
        throw std::invalid_argument("not a squared matrix");
    }
    if(O.rows()!=U.rows()||O.cols()!=U.cols()){
        O.resizeLike(U);
    }
    O.setZero();
    int n=U.rows();
    int i,j,k;
    for(i=1;i<=n;i++){
        for(j=1;j<=n;j++){
            T temp=0;
            for(k=i;k<=n;k++){
                temp+=U(k-1,i-1)*U(k-1,j-1);
            }
            O(i-1,j-1)=temp;
        }
    }
}

template<typename T, int _Options>
T Decomposition<T,_Options>::matrix3x3Determinant(const Decomposition::Matrix3x3T &A)
{
    const T* m=A.data();
    return m[0] * m[4] * m[8] + m[1] * m[5] * m[6] + m[2] * m[3] * m[7]
             - m[2] * m[4] * m[6] - m[1] * m[3] * m[8] - m[0] * m[5] * m[7];
}

template<typename T, int _Options>
typename Decomposition<T,_Options>::Matrix3x3T Decomposition<T,_Options>::matrix3x3Inverse(const Decomposition::Matrix3x3T &A, T det)
{

    Matrix3x3T O;
    T* ret=O.data();
    const T* m=A.data();
    ret[0] = m[4] * m[8] - m[5] * m[7];
    ret[1] = m[2] * m[7] - m[1] * m[8];
    ret[2] = m[1] * m[5] - m[2] * m[4];
    ret[3] = m[5] * m[6] - m[3] * m[8];
    ret[4] = m[0] * m[8] - m[2] * m[6];
    ret[5] = m[2] * m[3] - m[0] * m[5];
    ret[6] = m[3] * m[7] - m[4] * m[6];
    ret[7] = m[1] * m[6] - m[0] * m[7];
    ret[8] = m[0] * m[4] - m[1] * m[3];
    return O / det;
}
template<typename T, int _Options>
void Decomposition<T,_Options>::LLT(const Decomposition::MatrixXXT &A, Decomposition::MatrixXXT &L)
{
    if(A.rows()!=A.cols()){
        throw std::invalid_argument("not a squared matrix");
    }
    if(L.rows()!=A.rows()||L.cols()!=A.cols()){
        L.resizeLike(A);
    }
    L.setZero();
    T temp;
    int n=A.rows();
    int i,j,k;
    for(i=1;i<=n;i++){

        for(j=1;j<i;j++){
            temp=0;
            for(k=1;k<=j-1;k++){
                temp+=L(i-1,k-1)*L(j-1,k-1);
            }
            L(i-1,j-1)=(A(i-1,j-1)-temp)/L(j-1,j-1);
        }
        temp=0;
        for(k=1;k<=i-1;k++){
            temp+=L(i-1,k-1)*L(i-1,k-1);
        }
        L(i-1,i-1)=std::sqrt(A(i-1,i-1)-temp);
    }
}
template<typename T, int _Options>
void Decomposition<T,_Options>::LDLT(const MatrixXXT &A, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, _Options> &L,MatrixXXT& D)
{
    if(A.rows()!=A.cols()){
        throw std::invalid_argument("not a squared matrix");
    }
    if(L.rows()!=A.rows()||L.cols()!=A.cols()){
        L.resizeLike(A);
    }
    if(D.rows()!=A.rows()||D.cols()!=D.cols()){
        D.resizeLike(A);
    }
    L.setZero();
    D.setZero();
    int i,j,k;
    int n=A.rows();
    T temp;
    for(i=1;i<=n;i++){
        for(j=1;j<=i;j++){
            if(j==1)L(i-1,j-1)=A(i-1,j-1);
            else{
                temp=0;
                for(k=1;k<=j-1;k++){
                    temp+=L(i-1,k-1)*L(j-1,k-1)/L(k-1,k-1);
                }
                L(i-1,j-1)=A(i-1,j-1)-temp;
            }
        }
        D(i-1,i-1)=1/L(i-1,i-1);
    }
}
#endif // DECOMPOSITION_H

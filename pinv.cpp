#include <vector>
#include "mkl.h"
#include "mkl_lapacke.h"
#include <iostream>

using namespace std;

#define MINMN(M, N) ((M)>(N)?(N):(M))

void vec_ptr(double* vec, int n)
{
    for(int i=0;i<n;i++)
    {
        cout << vec[i] << ' ';
    }
    cout << endl;
}

void mat_ptr(double* A, int m, int n)
{
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            cout << A[i*n+j] << ' ';
        }
        cout << '\n';
    }
}

void diagonal_gen(double* a, int n, double* c, int k)
{
    int lda = n;
    for (int i = 0; i < k; i++)
    {
        a[i*lda+i] = c[i];
    }
}

void gesvd_driver(double* a, int m, int n, double* s, double* u, double* vt, double* superb, int* info)
{
    // u = m x m  , s = n, v = n x n
    int lda = n;
    int ldu = m;
    int ldvt = n;
    int info_local;
    info_local = LAPACKE_dgesvd( LAPACK_ROW_MAJOR, 'A', 'A', m, n, a, lda, s, u, ldu, vt, ldvt, superb );
    *info = info_local;
}

void vec_zero(double* A, int n)
{
    for (int i = 0; i < n; i++)
    {
        A[i] = 0.0;
    } 
}

void vec_inverse(double* s, int k)
{
    for(int i=0; i<k; i++)
    {
        if(abs(s[i]) > 1.0e-9)
            s[i]=1.0/s[i];
        else
            s[i]=s[i];
    }
}

void pinv_driver(double* A, int m, int n, int k, double* S, double* Smat, double* Sp, double* U, double* Vt, double* Superb, int* info_svd, double* USigma, double* Pinv)
{
    gesvd_driver(A, m, n, S, U, Vt, Superb, info_svd);
    vec_inverse(S, k);
    vec_zero(Smat, m*n);
    diagonal_gen(Smat, m, S, k);
    vec_zero(USigma, m*n);
    vec_zero(Pinv, m*n);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, m, n, 1.0, Vt, n, Smat, m, 0.0, USigma, m);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, m, 1.0, USigma, m, U, m, 0.0, Pinv, m);
}

void pinv_API(vector<double>& A_vec, int m, int n, vector<double>& Pinv_vec)
{
    int k = MINMN(m, n);
    double* U, *Vt, *S, *Superb, *Smat, *Sp, *USigma, *Pinv, *A;
    U = new double[m*m];
    S = new double[MINMN(m, n)];
    Vt = new double[n*n];
    Superb = new double[MINMN(m,n)-1];
    Smat = new double[m*n];
    Sp = new double[m*n];
    USigma = new double[m*n];
    Pinv = new double[m*n];
    A = new double[m*n];
    int info_svd;
    cblas_dcopy(m*n, &*A_vec.begin(), 1, A, 1);
    
    pinv_driver(A, m, n, k, S, Smat, Sp, U, Vt, Superb, &info_svd, USigma, Pinv);

    cblas_dcopy(m*n, Pinv, 1, &*Pinv_vec.begin(), 1);
}

int main(void)
{
    // int m = 3;
    // int n = 2;
    // int k = MINMN(m, n);
    // double* U, *Vt, *S, *Superb, *Smat, *Sp, *USigma, *Pinv;
    // U = new double[m*m];
    // S = new double[MINMN(m, n)];
    // Vt = new double[n*n];
    // Superb = new double[MINMN(m,n)-1];
    // Smat = new double[m*n];
    // Sp = new double[m*n];
    // USigma = new double[m*n];
    // Pinv = new double[m*n];
    // int info_svd;

    // double A[6] = {2, -1, 0, 4, 3, -2};

    // pinv_driver(A, m, n, k, S, Smat, Sp, U, Vt, Superb, &info_svd, USigma, Pinv);

    // mat_ptr(Pinv, n, m);

    // vector<double> A_vec {2, -1, 0, 4, 3, -2};
    // vector<double> Pinv_vec(6);

    // pinv_API(A_vec, 3, 2, Pinv_vec);
    // mat_ptr(&*Pinv_vec.begin(), 2, 3);

    vector<double> B {1,2,3,4,5,6,7,8};
    vector<double> Pinv(8);

    pinv_API(B, 2, 4, Pinv);
    mat_ptr(&*Pinv.begin(), 4, 2);

}
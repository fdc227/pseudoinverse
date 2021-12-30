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
    int lda = n+1;
    for (int i = 0; i < k; i++)
    {
        a[i*lda] = c[i];
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
        if(s[i] > 1.0e-9)
            s[i]=1.0/s[i];
        else
            s[i]=s[i];
    }
}

void pinv_driver(double* A, int m, int n, int k, double* S, double* Smat, double* Sp, double* U, double* Vt, double* Superb, int* info_svd, double* USigma, double* Pinv)
{
    gesvd_driver(A, m, n, S, U, Vt, Superb, info_svd);
    vec_inverse(S, k);
    diagonal_gen(Smat, n, S, k);
    mat_ptr(Smat, m, n);
    vec_zero(USigma, m*n);
    vec_zero(Pinv, m*n);
    // void cblas_dgemm (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT n, const MKL_INT k, const double alpha, const double *a, const MKL_INT lda, const double *b, const MKL_INT ldb, const double beta, double *c, const MKL_INT ldc);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, m, 1.0, U, m, Smat, n, 1.0, USigma, n);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, n, m, n, 1.0, Vt, n, USigma, n, 1.0, Pinv, m);
    
}

int main(void)
{
    int m = 2;
    int n = 3;
    int k = MINMN(m, n);
    double* U, *Vt, *S, *Superb, *Smat, *Sp, *USigma, *Pinv;
    U = new double[m*m];
    S = new double[MINMN(m, n)];
    Vt = new double[n*n];
    Superb = new double[MINMN(m,n)-1];
    Smat = new double[m*n];
    Sp = new double[m*n];
    USigma = new double[m*n];
    Pinv = new double[m*n];
    int info_svd;
    // double A[6*5] = 
    // {
    //     8.79,  9.93,  9.83, 5.45,  3.16,
    //     6.11,  6.91,  5.04, -0.27,  7.98,
    //     -9.15, -7.93,  4.86, 4.85,  3.01,
    //     9.57,  1.64,  8.83, 0.74,  5.80,
    //     -3.49,  4.02,  9.80, 10.00,  4.27,
    //     9.84,  0.15, -8.99, -6.02, -5.31
	// };
    double A[6] = {2, -1, 0, 4, 3, -2};

    pinv_driver(A, m, n, k, S, Smat, Sp, U, Vt, Superb, &info_svd, USigma, Pinv);

    mat_ptr(Pinv, n, m);

    // gesvd_driver(a, m, n, s, u, vt, superb, &info);

    // mat_ptr(u, m, m);
    // cout << endl;
    // mat_ptr(s, 1, n);
    // cout << endl;
    // mat_ptr(vt, n, n);
}



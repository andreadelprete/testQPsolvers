/*
 *	Test of QP solvers for performance comparison.
 *  @author Andrea Del Prete - LAAS/CNRS
 */

#include <testQPsolvers/testQPsolvers.hpp>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <iostream>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

using namespace Eigen;
using namespace std;

/** Solve the QP assuming always that the first avgActiveInequalities inequalities
 * are active, while the others are not. This way it is like the QP has only
 * equality constraints, which are solved by resolving the full KKT system.
 */
bool testEigenEq(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
                const vector<VectorXd> &gradientPerturbations, int avgActiveInequalities,
                double &avgTime,VectorXd &optCosts)
{
    /* Setting up QProblem object. */
    int n = qp.lb.size();
    int m = avgActiveInequalities;
    int nTest = hessianPerturbations.size();

    VectorXd xyOpt(n+m);

    VectorXd kkt(n+m);
    kkt.head(n) = -qp.g;
    kkt.tail(m) = -qp.ubA.head(m);

    MatrixXd KKT(n+m,n+m);
    KKT.topLeftCorner(n,n) = qp.H;
    KKT.topRightCorner(n,m) = qp.A.topRows(m).transpose();
    KKT.bottomLeftCorner(m,n) = qp.A.topRows(m);
    KKT.bottomRightCorner(m,m).setZero();
    // Eigen LDLT decomposition works only on PSD matrices, while the KKT matrix is only symmetric invertible
//    LDLT<MatrixXd> ldltOfKKT(KKT);
    PartialPivLU<MatrixXd> luOfKKT(KKT);

    clock_t t = clock();
    for(int i=0; i<nTest; i++)
	{
        kkt.head(n)            -= gradientPerturbations[i];
        KKT.topLeftCorner(n,n) += hessianPerturbations[i];
        luOfKKT.compute(KKT);
        xyOpt = luOfKKT.solve(kkt);
        optCosts[i] = 0.5*xyOpt.head(n).dot(KKT.topLeftCorner(n,n)*xyOpt.head(n))
                      - kkt.head(n).dot(xyOpt.head(n));
//        if(i<2)
//        {
//            cout<<"xyOpt "<<xyOpt.transpose()<<endl;
//            cout<<"kkt "<<kkt.transpose()<<endl;
//            cout<<"KKT\n"<<KKT<<endl;
////            MatrixXd L = ldltOfKKT.matrixL(); cout<<"L\n"<<L<<endl;
////            MatrixXd U = ldltOfKKT.matrixU(); cout<<"U\n"<<U<<endl;
////            VectorXd d = ldltOfKKT.vectorD(); cout<<"d: "<<d<<endl;
//            MatrixXd PLU = luOfKKT.reconstructedMatrix();
//            cout<<"PLU: "<<PLU<<endl;
//        }
	}
	t = clock() - t;

    avgTime = ((float)t)/(CLOCKS_PER_SEC*nTest);

    return true;
}

/** Solve the QP assuming always that the first avgActiveInequalities inequalities
 * are active, while the others are not. This way it is like the QP has only
 * equality constraints, which are solved by block elimination of the KKT system.
 */
bool testEigenEqElim(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
                const vector<VectorXd> &gradientPerturbations, int avgActiveInequalities,
                double &avgTime,VectorXd &optCosts)
{
    /* Setting up QProblem object. */
    int n = qp.lb.size();
    int m = avgActiveInequalities;
    int nTest = hessianPerturbations.size();

    VectorXd xOpt(n);               // primal solution
    VectorXd Hinv_g(n);
    VectorXd y(m);                  // Lagrange multipliers
    VectorXd g(n);                  // gradient
    MatrixXd H = qp.H;              // Hessian
    MatrixXd A = qp.A.topRows(m);   // linear equality constraint matrix
    VectorXd ubA = qp.ubA.head(m);
    MatrixXd Hinv_AT = A.transpose();
    MatrixXd S = -A*Hinv_AT;        // Schur complement
    LLT<MatrixXd> lltOfH(H);
    LLT<MatrixXd> lltOfS(S);

    clock_t t = clock();
    for(int i=0; i<nTest; i++)
	{
        // perturb the problem
        g  += gradientPerturbations[i];
        H  += hessianPerturbations[i];
        // compute H^-1*g
        lltOfH.compute(H);
        Hinv_g = g;
        lltOfH.solveInPlace(Hinv_g);
        // compute H^-1*A^T
        lltOfH.solveInPlace(Hinv_AT);
        // solve S*y = A*H^-1*g - b
        S = -A*Hinv_AT;
        lltOfS.compute(S);
        y = Hinv_AT.transpose()*g - ubA;
        lltOfS.solveInPlace(y);
        // solve H*x = -A^T*y - g
        xOpt = -A.transpose()*y - g;
        lltOfH.solveInPlace(xOpt);

        optCosts[i] = 0.5*xOpt.dot(H*xOpt) + g.dot(xOpt);
	}
	t = clock() - t;

    avgTime = ((float)t)/(CLOCKS_PER_SEC*nTest);

    return true;
}

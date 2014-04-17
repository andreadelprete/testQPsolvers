/*
 *	Test of QP solvers for performance comparison.
 *  @author Andrea Del Prete - LAAS/CNRS
 */

#ifndef TESTQPSOLVER_TESTQPSOLVER_H
#define TESTQPSOLVER_TESTQPSOLVER_H

#include <Eigen/Core>
#include <vector>

using namespace Eigen;
using namespace std;

// define a type for the matrix stored in rowMajor
typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;

// minimize    0.5*x^T*H*x + g^T*x
// subject to
//             A*x <= ubA
//             lb  <=  x  <= ub
struct QPdata
{
    MatrixXdr H;    // Hessian
    MatrixXdr A;    // Constraint matrix
    VectorXd g;     // gradient
    VectorXd lb;    // box constraint lower bounds
    VectorXd ub;    // box constraint upper bounds
    VectorXd ubA;   // constraint upper bounds
    int m_box;      // number of box constraints
};

#define NO_UPPER_BOUND 10e12
#define NO_LOWER_BOUND -10e12

void orderLowerUpperBounds(VectorXd &lb, VectorXd &ub);


bool testQpOases(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
                 const vector<VectorXd> &gradientPerturbations, bool useWarmStart,
                 double &avgTime, double &avgWSR, double &avgActiveConstr, VectorXd &optCosts);

bool testOOQPDense(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
               const vector<VectorXd> &gradientPerturbations,
               double &avgTime, double &avgWSR, VectorXd &optCosts);

bool testOOQPSparse(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
               const vector<VectorXd> &gradientPerturbations,
               double &avgTime, double &avgWSR, VectorXd &optCosts);

bool testQuadProgPP(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
                    const vector<VectorXd> &gradientPerturbations,
                    double &avgTime, double &avgWSR, VectorXd &optCosts);

bool testMosek(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
                 const vector<VectorXd> &gradientPerturbations, double &avgTime, VectorXd &optCosts);

bool testEigenEq(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
                 const vector<VectorXd> &gradientPerturbations, int avgActiveInequalities,
                 double &avgTime, VectorXd &optCosts);

bool testEigenEqElim(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
                const vector<VectorXd> &gradientPerturbations, int avgActiveInequalities,
                double &avgTime,VectorXd &optCosts);



#endif

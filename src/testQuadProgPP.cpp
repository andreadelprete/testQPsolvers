/*
 *	Test of QP solvers for performance comparison.
 *  @author Andrea Del Prete - LAAS/CNRS
 */

#include <testQPsolvers/testQPsolvers.hpp>
#include <testQPsolvers/eiquadprog.hpp>
#include <iostream>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

using namespace Eigen;
using namespace std;

bool testQuadProgPP(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
    const vector<VectorXd> &gradientPerturbations,
    double &avgTime, double &avgWSR, VectorXd &optCosts)
{
    /* Setting up problem variables */
    int n = qp.lb.size();
    int m = qp.ubA.size();
    int m_box = qp.m_box;
    int nTest = hessianPerturbations.size();

    VectorXd gTemp = qp.g;
    MatrixXd HTemp = qp.H;
    MatrixXd HTemp2 = qp.H;    // to copy the Hessian because it is modified within the function
    MatrixXd CE(0,0);                 // no equality constraints
    VectorXd ce0;
    VectorXd sol(n);
    MatrixXd CIT(m, n);
    VectorXd ci0(m);

    // convert box constraints into standard inequality constraints
    CIT.topRows(m) = -1*qp.A; // negate constraint matrix to match the form CI^T x + ci0 >= 0
    //CI.block(m,0,m_box,m_box) = -1.0*MatrixXd::Identity(m_box,m_box);
    //CI.block(m+m_box,0,m_box,m_box) = MatrixXd::Identity(m_box,m_box);
    MatrixXd CI = CIT.transpose();
    ci0.head(m) = qp.ubA;
    //ci0.segment(m,m_box) = qp.ub.head(m_box);
    //ci0.tail(m_box) = -1.0*qp.lb.head(m_box);

    clock_t t = clock();
    for(int i=0; i<nTest; i++)
	{
        gTemp += gradientPerturbations[i];
        HTemp += hessianPerturbations[i];

        // Copy the Hessian because it is modified within the function
        HTemp2 = HTemp;

        //min  0.5 * x H x + g x
        //s.t. CE^T x + ce0 = 0
        //     CI^T x + ci0 >= 0
        // Since you cannot explicitly express bilateral constraints
        // consider only lower bounds
        optCosts[i] = solve_quadprog(HTemp2, gTemp, CE, ce0, CI, ci0, sol);

        if(optCosts[i]==std::numeric_limits<double>::infinity())
            cout<<"QP "<<i<<" failed.\n";
	}
	t = clock() - t;

    avgTime = ((float)t)/(CLOCKS_PER_SEC*nTest);
    return true;
}

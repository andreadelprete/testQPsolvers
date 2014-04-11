/*
 *	Test of QP solvers for performance comparison.
 *  @author Andrea Del Prete - LAAS/CNRS
 */

#include <testQPsolvers/testQPsolvers.hpp>
#include "QpGenData.h"
#include "QpGenVars.h"
#include "QpGenResiduals.h"
#include "GondzioSolver.h"
//#include "QpGenSparseMa27.h"
#include "QpGenDense.h"
#include <iostream>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

using namespace Eigen;
using namespace std;

typedef Matrix<char,Dynamic,1> VectorXc;

bool testOOQP(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
              const vector<VectorXd> &gradientPerturbations,
              double &avgTime, double &avgWSR, VectorXd &optCosts)
{
    /* Setting up QProblem object. */
    int n = qp.lb.size();
    int m = qp.ubA.size();
    int nTest = hessianPerturbations.size();
    int nWSR;
    avgWSR = 0.0;
    VectorXd g = qp.g;
    MatrixXdr H = qp.H;
    MatrixXdr A = qp.A;
    VectorXd ubA = qp.ubA;
    VectorXd xlow = VectorXd::Zero(n);
    VectorXc ixlow = VectorXc::Zero(n);
    VectorXd xupp = VectorXd::Zero(n);
    VectorXc ixupp = VectorXc::Zero(n);

    VectorXd clow = VectorXd::Zero(m);
    VectorXc iclow(m);
    //char *iclow = new char[m];
    VectorXc icupp(m);

    for(int i=0;i<m;i++)
    {
        iclow[i] = 0;
        icupp[i] = 1;
    }

    QpGenDense *qpoo = new QpGenDense( n, 0, m);
    QpGenData *prob = (QpGenData*) qpoo->makeData(
        g.data(), H.data(),  /*cost function*/
        xlow.data(), ixlow.data(), xupp.data(), ixupp.data(), /*box bounds*/
        NULL, NULL,  /*equality constraints*/
        A.data(), clow.data(), iclow.data(), ubA.data(), icupp.data() /*inequality constraints*/
        );
    QpGenVars *vars = (QpGenVars*) qpoo->makeVariables(prob);
    QpGenResiduals *resid = (QpGenResiduals*) qpoo->makeResiduals(prob);
    GondzioSolver *s = new GondzioSolver(qpoo, prob);
//    Map<VectorXd> xOpt(vars->x.ptr()->copyIntoArray,n);
    VectorXd xOpt(n);
    //s->monitorSelf();

    clock_t t = clock();
    for(int i=0; i<nTest; i++)
	{
        g += gradientPerturbations[i];
        H += hessianPerturbations[i];

        int res = s->solve(prob, vars, resid);

        if(res!=0)
            cout<<"QP "<<i<<" failed\n";
        else
        {
            vars->x.ptr()->copyIntoArray(xOpt.data());
            optCosts[i] = 0.5*xOpt.dot(H*xOpt) + g.dot(xOpt);
        }
	}
	t = clock() - t;

    avgTime = ((float)t)/(CLOCKS_PER_SEC*nTest);
    return true;
}

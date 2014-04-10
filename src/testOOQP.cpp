/*
 *	Test of QP solvers for performance comparison.
 *  @author Andrea Del Prete - LAAS/CNRS
 */

#include <testQPsolvers/testQPsolvers.hpp>
#include <iostream>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

using namespace Eigen;
using namespace std;

bool testOOQP(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
              const vector<VectorXd> &gradientPerturbations, bool useWarmStart,
              double &avgTime, double &avgWSR, VectorXd &optCosts)
{
    /* Setting up QProblem object. */
    int n = qp.lb.size();
    int m = qp.ubA.size();
    int nTest = hessianPerturbations.size();
    int nWSR;
    avgWSR = 0.0;
    VectorXd gTemp = qp.g;
    MatrixXd HTemp = qp.H;

    clock_t t = clock();
    for(int i=0; i<nTest; i++)
	{
        gTemp += gradientPerturbations[i];
        HTemp += hessianPerturbations[i];

        nWSR = 2*(m+n);
        avgWSR += nWSR;

        if(false)
        {
            cout<<"QP "<<i<<" failed\n";
        }
	}
	t = clock() - t;

    avgTime = ((float)t)/(CLOCKS_PER_SEC*nTest);
    avgWSR /= nTest;
    return true;
}

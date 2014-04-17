/*
 *	Test of QP solvers for performance comparison.
 *  @author Andrea Del Prete - LAAS/CNRS
 */

#include <testQPsolvers/testQPsolvers.hpp>
#include "QpGenData.h"
#include "QpGenVars.h"
#include "QpGenResiduals.h"
#include "GondzioSolver.h"
#include "QpGenSparseMa27.h"
#include "QpGenDense.h"
#include <iostream>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

using namespace Eigen;
using namespace std;

typedef Matrix<char,Dynamic,1> VectorXc;

bool testOOQPDense(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
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


bool testOOQPSparse(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
              const vector<VectorXd> &gradientPerturbations,
              double &avgTime, double &avgWSR, VectorXd &optCosts)
{
    /* Setting up QProblem object. */
    int n = qp.lb.size();
    int m = qp.ubA.size();
    int nTest = hessianPerturbations.size();
    VectorXd g = qp.g;
    MatrixXdr H = qp.H;
    MatrixXdr A = qp.A;
    VectorXd ubA = qp.ubA;
    VectorXd xlow = VectorXd::Zero(n);
    VectorXc ixlow = VectorXc::Zero(n);
    VectorXd xupp = VectorXd::Zero(n);
    VectorXc ixupp = VectorXc::Zero(n);
    VectorXd clow = VectorXd::Zero(m);
    VectorXc iclow(m), icupp(m);

    for(int i=0;i<m;i++)
    {
        iclow[i] = 0;
        icupp[i] = 1;
    }

    int nt = n*(n+1)/2; // num of elements in triangular n-by-n matrix
    VectorXd Hsparse(nt);    // sparse representation of H
    VectorXd Asparse(n*m);   // sparse representation of A
    VectorXi Hrowi(nt), Hcolj(nt), Arowi(n*m), Acolj(n*m);
    int indH=0, indA=0;
    for(int i=0; i<n; ++i)
        for(int j=0; j<=i; j++)
        {   // save H in a vector containing only the lower triangular part
            Hsparse[indH] = H(i,j);
            Hrowi(indH) = i;
            Hcolj(indH) = j;
            indH++;
        }

    for(int i=0; i<m; ++i)
        for(int j=0; j<n; j++)
        {
            Asparse(indA) = A(i,j);
            Arowi(indA) = i;
            Acolj(indA) = j;
            indA++;
        }

    cout<<"OOQP sparse - so far so good\n";
    QpGenSparseMa27 *qpoo = new QpGenSparseMa27(n, 0, m, nt, 0, m*n);
    cout<<"OOQP sparse - created QpGenSparseMa27\n";
    cout<<"Gradient: "<<g.transpose()<<endl;
    cout<<"Hrowi: "<<Hrowi.transpose()<<endl;
    cout<<"Hcolj: "<<Hcolj.transpose()<<endl;
    //cout<<"Hsparse: "<<Hsparse.transpose()<<endl;
    cout<<"Arowi: "<<Arowi.transpose()<<endl;
    cout<<"Acolj: "<<Acolj.transpose()<<endl;
    //cout<<"Asparse: "<<Asparse.transpose()<<endl;
    QpGenData *prob = NULL;
//    (QpGenData*) qpoo->makeData(
//        g.data(), Hrowi.data(), Hcolj.data(), Hsparse.data(),  /*cost function*/
//        xlow.data(), ixlow.data(), xupp.data(), ixupp.data(), /*box bounds*/
//        NULL, NULL, NULL, NULL,  /*equality constraints*/
//        Arowi.data(), Acolj.data(), Asparse.data(),
//        clow.data(), iclow.data(), ubA.data(), icupp.data() /*inequality constraints*/
//        );
    cout<<"OOQP sparse - makeData ok\n";
    QpGenVars *vars = (QpGenVars*) qpoo->makeVariables(prob);
    QpGenResiduals *resid = (QpGenResiduals*) qpoo->makeResiduals(prob);
    GondzioSolver *s = new GondzioSolver(qpoo, prob);
    VectorXd xOpt(n);
    //s->monitorSelf();

    cout<<"Starting testing OOQP sparse\n";
    clock_t t = clock();
    int i=0;
    cout<<"clocked\n";
    for(; i<nTest; i++)
	{
        cout<<"Perturb g ";
        g += gradientPerturbations[i];
        cout<<"Save hessian perturbation ";
        const MatrixXd &hp = hessianPerturbations[i];
        indH=0;
        cout<<"Perturb hessian ";
        for(int ii=0; ii<n; ++ii)
            for(int j=0; j<=ii; j++)
            {   // save H in a vector containing only the lower triangular part
                Hsparse[indH] += hp(ii,j);
                indH++;
            }

        cout<<"\tStart QP "<<i;
        int res = s->solve(prob, vars, resid);

        if(res!=0)
            cout<<"QP "<<i<<" failed\n";
        else
        {
            vars->x->copyIntoArray(xOpt.data());
            optCosts[i] = 0.5*xOpt.dot(H*xOpt) + g.dot(xOpt);
        }
	}
	t = clock() - t;

    avgTime = ((float)t)/(CLOCKS_PER_SEC*nTest);
    return true;
}

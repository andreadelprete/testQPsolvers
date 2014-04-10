/*
 *	Test of QP solvers for performance comparison.
 *  @author Andrea Del Prete - LAAS/CNRS
 */

#include <testQPsolvers/testQPsolvers.hpp>
#include <qpOASES.hpp>
#include <iostream>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

USING_NAMESPACE_QPOASES
using namespace Eigen;
using namespace std;

bool testQpOases(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
                const vector<VectorXd> &gradientPerturbations, bool useWarmStart,
                double &avgTime, double &avgWSR, double &avgActiveConstr, VectorXd &optCosts)
{
    /* Setting up QProblem object. */
    int n = qp.lb.size();
    int m = qp.ubA.size();
    int nTest = hessianPerturbations.size();

    SQProblem solverQPoases(n,m);
    Options options;
    options.initialStatusBounds = ST_INACTIVE;
    options.printLevel          = PL_NONE;
    solverQPoases.setOptions( options );
	//example.printOptions();

    /* Solve first QP. */
    int nWSR = (m+n)*2;
    int res = solverQPoases.init(qp.H.data(), qp.g.data(), qp.A.data(), qp.lb.data(), qp.ub.data(), NULL, qp.ubA.data(), nWSR);
    if(res!=SUCCESSFUL_RETURN)
    {
        cout<<"First QP failed with code "<<res<<": ";
        if(res==RET_MAX_NWSR_REACHED) cout<<"MAX_NWSR_REACHED"<<endl;
        else if(res==RET_HOTSTART_STOPPED_INFEASIBILITY) cout<<"RET_HOTSTART_STOPPED_INFEASIBILITY"<<endl;
        else cout<<"UNKNOWN"<<endl;
        return false;
    }

    /* Get and print solution of first QP. */
//    VectorXd xOpt(n), yOpt(n+m);
//    solverQPoases.getPrimalSolution( xOpt.data() );
//    solverQPoases.getDualSolution( yOpt.data() );

//        cout<<"Hessian matrix:\n"<<H<<endl;
//        cout<<"Constraint matrix:\n"<<A<<endl;
//        cout<<"Gradient: "<<g.transpose()<<endl;
//        cout<<"Unconstrained optimum: "<<x_opt_unc.transpose()<<endl;
//        cout<<"Lower bounds: "<<lb.transpose()<<endl;
//        cout<<"Upper bounds: "<<ub.transpose()<<endl;
//        cout<<"Constraint unc opt value: "<<constrVal.transpose()<<endl;
//        cout<<"Constraint Lower bounds: "<<lbA.transpose()<<endl;
//        cout<<"Constraint Upper bounds: "<<ubA.transpose()<<endl;
//        cout<<"\nxOpt = "<<xOpt.transpose()<<endl;
//        cout<<"yOpt = "<<yOpt.transpose()<<endl;
//        cout<<"objVal = "<<solverQPoases.getObjVal()<<"\n\n";

    /* Solve subsequent QPs */
    avgWSR = 0.0;
    avgActiveConstr = 0.0;
    int activeConstr = 0;
    VectorXd yOpt(m+n);
    VectorXd gTemp = qp.g;
    MatrixXd HTemp = qp.H;

    clock_t t = clock();
    for(int i=0; i<nTest; i++)
	{
        gTemp += gradientPerturbations[i];
        HTemp += hessianPerturbations[i];

        nWSR = 2*(m+n);
        if(useWarmStart)
            res = solverQPoases.hotstart(HTemp.data(),gTemp.data(), qp.A.data(), qp.lb.data(), qp.ub.data(), NULL, qp.ubA.data(), nWSR);
        else
            res = solverQPoases.init(HTemp.data(),gTemp.data(), qp.A.data(), qp.lb.data(), qp.ub.data(), NULL, qp.ubA.data(), nWSR);
        avgWSR += nWSR;

        if(res!=SUCCESSFUL_RETURN)
        {
            cout<<"QP "<<i<<" failed with code "<<res<<": ";
            if(res==RET_MAX_NWSR_REACHED)
                cout<<"MAX_NWSR_REACHED"<<endl;
            else if(res==RET_HOTSTART_STOPPED_INFEASIBILITY)
                cout<<"RET_HOTSTART_STOPPED_INFEASIBILITY"<<endl;
            else
                cout<<endl;
            solverQPoases.reset();
        }
        else
        {
            optCosts[i] = solverQPoases.getObjVal();
            solverQPoases.getDualSolution(yOpt.data());
            for(int j=0; j<m+n; j++)
                if(yOpt[j]!=0.0)
                    avgActiveConstr += 1;
            //cout<<"QP "<<i<<" solved in "<<nWSR<<" iterations\n";
        }
	}
	t = clock() - t;

    avgTime = ((float)t)/(CLOCKS_PER_SEC*nTest);
    avgWSR /= nTest;
    avgActiveConstr /= nTest;
    return true;
}

/*
 *	Test of QP solvers for performance comparison.
 *  @author Andrea Del Prete - LAAS/CNRS
 */

#include <qpOASES.hpp>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <iostream>

USING_NAMESPACE_QPOASES
using namespace Eigen;
using namespace std;

typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;

void orderLowerUpperBounds(VectorXd &lb, VectorXd &ub)
{
    int n = lb.size();
    for(int i=0; i<n; i++)
    {
        if(lb[i]>=ub[i])    // if lower bound greater than upper bound, swap them
        {
            double temp = lb[i];
            lb[i] = ub[i];
            ub[i] = temp;
        }
    }
}

int main( )
{
	/* Setup data of first QP. */
	const int nTest = 3;   // number of tests to execute
    const int n     = 2;    // number of variables
    const int m     = n*1;  // number of constraints
    int nWSR         = m*100;    // max number of active set update
    double NORMAL_DISTR_VAR = 100.0;
    double MARGIN_PERC = 0.01;

    MatrixXdr pippo = MatrixXdr::Random(n,n);

    for(int i=0; i<nTest; i++)
	{
        // minimize    x^T*H*x + g^T*x
        // subject to
        //             lbA <= A*x <= ubA
        //             lb  <=  x  <= ub
        MatrixXdr H = MatrixXdr::Random(n,n);
        H = H * H.transpose();
        MatrixXdr A = MatrixXdr::Random(m,n);
        VectorXd g = VectorXd::Random(n);
        VectorXd lb = VectorXd::Random(n)*NORMAL_DISTR_VAR;
        VectorXd ub = VectorXd::Random(n)*NORMAL_DISTR_VAR;
        VectorXd lbA = VectorXd::Random(m)*NORMAL_DISTR_VAR;
        VectorXd ubA = VectorXd::Random(m)*NORMAL_DISTR_VAR;

        orderLowerUpperBounds(lb, ub);
        orderLowerUpperBounds(lbA, ubA);

        // compute unconstrained optimum
        VectorXd x_opt_unc = H.llt().solve(-1.0*g);
        VectorXd constrVal = A*x_opt_unc;
        for(int i=0; i<m; i++)
        {
            if(constrVal[i]<lbA[i]) lbA[i]=constrVal[i]-MARGIN_PERC*fabs(constrVal[i]);
            if(constrVal[i]>ubA[i]) ubA[i]=constrVal[i]+MARGIN_PERC*fabs(constrVal[i]);
        }
        for(int i=0; i<n; i++)
        {
            if(x_opt_unc[i]<lb[i])  lb[i] =x_opt_unc[i]-MARGIN_PERC*fabs(x_opt_unc[i]);
            if(x_opt_unc[i]>ub[i])  ub[i] =x_opt_unc[i]+MARGIN_PERC*fabs(x_opt_unc[i]);
        }

        VectorXd xOpt(n);
        VectorXd yOpt(n+m);

        cout<<"Hessian matrix:\n"<<H<<endl;
        cout<<"Constraint matrix:\n"<<A<<endl;
        cout<<"Gradient: "<<g.transpose()<<endl;
        cout<<"Lower bounds: "<<lb.transpose()<<endl;
        cout<<"Upper bounds: "<<ub.transpose()<<endl;
        cout<<"Constraint Lower bounds: "<<lbA.transpose()<<endl;
        cout<<"Constraint Upper bounds: "<<ubA.transpose()<<endl;
        cout<<"Unconstrained optimum: "<<x_opt_unc.transpose()<<endl;

        /* Setting up QProblem object. */
        QProblem solverQPoases(n,m);

        Options options;
        options.initialStatusBounds = ST_INACTIVE;
        options.printLevel          = PL_NONE;
        solverQPoases.setOptions( options );

        /* Solve first QP. */
        int res = solverQPoases.init(H.data(), g.data(), A.data(), lb.data(), ub.data(), lbA.data(), ubA.data(), nWSR);
        if(res!=SUCCESSFUL_RETURN)
        {
            cout<<"First QP failed with code "<<res<<": ";
            if(res==RET_MAX_NWSR_REACHED)
                cout<<"MAX_NWSR_REACHED"<<endl;
            else if(res==RET_HOTSTART_STOPPED_INFEASIBILITY)
                cout<<"RET_HOTSTART_STOPPED_INFEASIBILITY"<<endl;
            else
                cout<<"UNKNOWN"<<endl;
        }

        /* Get and print solution of first QP. */
        solverQPoases.getPrimalSolution( xOpt.data() );
        solverQPoases.getDualSolution( yOpt.data() );
        cout<<"\nxOpt = "<<xOpt.transpose()<<endl;
        cout<<"yOpt = "<<yOpt.transpose()<<endl;
        cout<<"objVal = "<<solverQPoases.getObjVal()<<"\n\n";

        /* Solve subsequent QPs */

        //cout<<"g pre: "<<g.transpose()<<endl;
        //g += VectorXd::Random(n)*0.000;
        //cout<<"g post: "<<g.transpose()<<endl;
        cout<<"Hessian matrix:\n"<<H<<endl;
        cout<<"Constraint matrix:\n"<<A<<endl;
        cout<<"Gradient: "<<g.transpose()<<endl;
        cout<<"Lower bounds: "<<lb.transpose()<<endl;
        cout<<"Upper bounds: "<<ub.transpose()<<endl;
        cout<<"Constraint Lower bounds: "<<lbA.transpose()<<endl;
        cout<<"Constraint Upper bounds: "<<ubA.transpose()<<endl;
        res = solverQPoases.init(H.data(),g.data(), A.data(), lb.data(), ub.data(), lbA.data(), ubA.data(), nWSR);
        //int res = solverQPoases.hotstart(g.data(), lb.data(), ub.data(), lbA.data(), ubA.data(), nWSR);
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
	}

	//example.printOptions();
	return 0;
}


/*
 *	end of file
 */

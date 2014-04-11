/*
 *	Test of QP solvers for performance comparison.
 *  @author Andrea Del Prete - LAAS/CNRS
 */

#include <testQPsolvers/testQPsolvers.hpp>
#include <qpOASES.hpp>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <iostream>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <vector>

USING_NAMESPACE_QPOASES
using namespace Eigen;
using namespace std;

int main( )
{
	/* Setup data of first QP. */
	const int nTest = 1000;  // number of tests to execute
    const int n     = 30;    // number of variables x
    const int m     = 34;    // number of linear constraints A*x <= u
    const int m_box = 0;     // number of box constraints    l <= x <= u
    // variance of the normal distribution used for generating initial bounds
    const double NORMAL_DISTR_VAR = 10.0;
    // each cycle the gradient is perturbed by a Gaussian random variable with this covariance
    const double GRADIENT_PERTURBATION_VARIANCE = 1e-2;
    // each cycle the Hessian is perturbed by a Gaussian random variable with this covariance
    const double HESSIAN_PERTURBATION_VARIANCE = 1e3;
    // min margin of activation of the inequality constraints for the first QP
    const double MARGIN_PERC = 1e-3;
    // if true then it uses the warm start
    const bool USE_WARM_START = true;
    bool res;

    QPdata initialQP;
    initialQP.H = MatrixXdr::Random(n,n);
    initialQP.H = initialQP.H * initialQP.H.transpose();
    initialQP.A = MatrixXdr::Random(m,n);
    initialQP.g = VectorXd::Random(n);
    initialQP.lb = VectorXd::Random(n)*NORMAL_DISTR_VAR;
    initialQP.ub = VectorXd::Random(n)*NORMAL_DISTR_VAR;
    initialQP.ubA = VectorXd::Random(m)*NORMAL_DISTR_VAR;
    initialQP.m_box = m_box;

    orderLowerUpperBounds(initialQP.lb, initialQP.ub);
    for(int i=m_box; i<n; i++)
    {
        initialQP.lb[i] = NO_LOWER_BOUND;
        initialQP.ub[i] = NO_UPPER_BOUND;
    }

    // compute unconstrained optimum and update bounds so that unconstrained optimum respect them
    VectorXd x_opt_unc = initialQP.H.llt().solve(-1.0*initialQP.g);
    VectorXd constrVal = initialQP.A*x_opt_unc;
    for(int i=0; i<m; i++)
    {
        if(constrVal[i]>initialQP.ubA[i]) initialQP.ubA[i]=constrVal[i]+MARGIN_PERC*fabs(constrVal[i]);
    }
    for(int i=0; i<n; i++)
    {
        if(x_opt_unc[i]<initialQP.lb[i])  initialQP.lb[i] =x_opt_unc[i]-MARGIN_PERC*fabs(x_opt_unc[i]);
        if(x_opt_unc[i]>initialQP.ub[i])  initialQP.ub[i] =x_opt_unc[i]+MARGIN_PERC*fabs(x_opt_unc[i]);
    }

    // Prepare random data to perturb initial QP
    vector<VectorXd> gradientPerturbations(nTest);
    vector<MatrixXd> hessianPerturbations(nTest);
    MatrixXd deltaH(n,n);
    for(int i=0; i<nTest; i++)
	{
        gradientPerturbations[i] = VectorXd::Random(n)*GRADIENT_PERTURBATION_VARIANCE;
        deltaH.setRandom(n,n);
        hessianPerturbations[i] = deltaH*deltaH.transpose()*HESSIAN_PERTURBATION_VARIANCE;
    }


    // TEST OOQP
    cout<<"\n*** TEST OOQP ("<<nTest<<" QP, "<<n<<" var, "<<m<<" unilateral inequality constraints)\n";
    double avgTime_OOQP, avgWSR_OOQP;
    VectorXd optCosts_OOQP(nTest);
    res = testOOQP(initialQP, hessianPerturbations, gradientPerturbations,
                              avgTime_OOQP, avgWSR_OOQP, optCosts_OOQP);
    if(res==false)
        cout<<"Test OOQP failed for some reason\n";
    cout<<"Average time for QP: "<<avgTime_OOQP*1000<<" ms\n";


    // TEST QUAD PROG++
    cout<<"\n*** TEST QUADPROG++ ("<<nTest<<" QP, "<<n<<" var, "<<m<<" unilateral inequality constraints)\n";
    double avgTime_quadProgPP, avgWSR_quadProgPP;
    VectorXd optCosts_quadProgPP(nTest);
    res = testQuadProgPP(initialQP, hessianPerturbations, gradientPerturbations,
                              avgTime_quadProgPP, avgWSR_quadProgPP, optCosts_quadProgPP);
    if(res==false)
        cout<<"Test QuadProg++ failed for some reason\n";
    cout<<"Average time for QP: "<<avgTime_quadProgPP*1000<<" ms\n";



    // TEST QP OASES
    cout<<"\n*** TEST QP OASES ("<<nTest<<" QP, "<<n<<" var, "<<m<<" bilateral inequality constraints)\n";
    double avgTime_qpoases, avgWSR_qpoases, avgActiveConstr_qpoases;
    VectorXd optCosts_qpoases(nTest);
    res = testQpOases(initialQP, hessianPerturbations, gradientPerturbations, USE_WARM_START,
                      avgTime_qpoases, avgWSR_qpoases, avgActiveConstr_qpoases, optCosts_qpoases);
    if(res==false)
        cout<<"Test QP oases failed for some reason\n";
    cout<<"Average time for QP: "<<avgTime_qpoases*1000<<" ms\n";
	cout<<"Average number of active set reconstructions: "<<avgWSR_qpoases<<endl;
	cout<<"Average number of active constraints: "<<avgActiveConstr_qpoases<<endl;


    cout<<endl;
	for(int i=0; i<5; i++)
	{
        cout<<"Optimal costs: ";
        cout<<optCosts_OOQP[i]<<" (OOQP)";
        cout<<optCosts_quadProgPP[i]<<" (qp++), ";
        cout<<optCosts_qpoases[i]<<" (qpoases)\n";
	}
}


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


/*
 *	end of file
 */

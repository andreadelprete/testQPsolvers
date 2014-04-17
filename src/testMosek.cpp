/*
 *	Test of QP solvers for performance comparison.
 *  @author Andrea Del Prete - LAAS/CNRS
 */

#include <testQPsolvers/testQPsolvers.hpp>
#include <mosek.h>
#include <iostream>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

using namespace Eigen;
using namespace std;

bool testMosek(const QPdata &qp, const vector<MatrixXd> &hessianPerturbations,
                const vector<VectorXd> &gradientPerturbations,
                double &avgTime, VectorXd &optCosts)
{
    /* Setting up QProblem object. */
    int n = qp.lb.size();
    int m = qp.ubA.size();
    int nTest = hessianPerturbations.size();

    VectorXd xOpt(n);
    VectorXd gTemp = qp.g;
    MatrixXd HTemp = qp.H;

    MSKenv_t env = NULL;
    MSKrescodee r = MSK_makeenv(&env, NULL);    /* Create an environment */

    MSKtask_t task = NULL;
    if (r == MSK_RES_OK)    /* Create a task */
        r = MSK_maketask(env, 0,0, &task);

    if ( r == MSK_RES_OK ) /* Append 'm' empty constraints (i.e. with no bounds). */
        r = MSK_appendcons(task, m);

    if ( r == MSK_RES_OK )  /* Append 'n' variables (initially fixed at 0). */
        r = MSK_appendvars(task, n);

    /* Set the linear term c_j in the objective.*/
    VectorXi aptrb(n);   // vector [0 1 2 ... n-1]
    int nt = n*(n+1)/2; // num of elements in triangular n-by-n matrix
    VectorXd Hsparse(nt);   // sparse representation of H
    VectorXi Hsubi(nt), Hsubj(nt);
    int ind=0;
    for(int i=0; i<n && r == MSK_RES_OK; ++i)
    {
        aptrb[i] = i;
        if(r == MSK_RES_OK) r = MSK_putcj(task,i,qp.g[i]);
        if(r == MSK_RES_OK) /* Set the bounds on variable i */
            r = MSK_putvarbound(task, i, MSK_BK_FR, -MSK_INFINITY, MSK_INFINITY);
        for(int j=0; j<=i; j++)
        {   // save H in a vector containing only the lower triangular part
            Hsparse[ind] = HTemp(i,j);
            Hsubi(ind) = i;
            Hsubj(ind) = j;
            ind++;
        }
    }

    /* Set constraints */
    for(int i=0; i<m && r==MSK_RES_OK; ++i)
    {
         /* Input row i of A */
        if(r == MSK_RES_OK)
            r = MSK_putarow(task, i, n,         /* Number of non-zeros in row i.*/
                            aptrb.data(),       /* Pointer to column indexes of row i.*/
                            qp.A.data()+i*n);   /* Pointer to values of row i.*/
        if(r == MSK_RES_OK)
            r = MSK_putconbound(task, i, MSK_BK_UP, /* Bound key.*/
                                -MSK_INFINITY,  /* Numerical value of lower bound.*/
                                qp.ubA[i]);     /* Numerical value of upper bound.*/
    }

    if ( r==MSK_RES_OK ) /* Set the lower triangular part of the Q matrix */
        r = MSK_putqobj(task,nt,Hsubi.data(),Hsubj.data(),Hsparse.data());

    if ( r==MSK_RES_OK )
        r = MSK_putobjsense(task, MSK_OBJECTIVE_SENSE_MINIMIZE);

    if ( r==MSK_RES_OK )    // turn off presolve step to save time
        r = MSK_putintparam(task, MSK_IPAR_PRESOLVE_USE, MSK_PRESOLVE_MODE_OFF);

    if ( r==MSK_RES_OK )    // Relative complementarity gap tolerance feasibility tolerance (def 1e-8)
        r = MSK_putdouparam(task,  MSK_DPAR_INTPNT_CO_TOL_MU_RED, 1e-1); // it seems not to affect the solver

    MSKsolstae solsta;
    clock_t t = clock();
    for(int i=0; i<nTest; i++)
	{
        // update gradient and Hessian
        gTemp += gradientPerturbations[i];
        const MatrixXd &hp = hessianPerturbations[i];
        ind=0;
        for(int ii=0; ii<n && r == MSK_RES_OK; ++ii)
        {
            if(r == MSK_RES_OK) r = MSK_putcj(task,ii,gTemp[ii]);
            for(int j=0; j<=ii; j++)
            {   // save H in a vector containing only the lower triangular part
                Hsparse[ind] += hp(ii,j);
                ind++;
            }
        }
        if ( r==MSK_RES_OK ) /* Set the lower triangular part of the Q matrix */
            r = MSK_putqobj(task,nt,Hsubi.data(),Hsubj.data(),Hsparse.data());

        // solve updated problem
        if ( r==MSK_RES_OK )
        {
            MSKrescodee trmcode;
            r = MSK_optimizetrm(task,&trmcode);     /* Run optimizer */
            //MSK_solutionsummary (task,MSK_STREAM_MSG);
            if ( r==MSK_RES_OK )
            {
                MSK_getprimalobj(task, MSK_SOL_ITR, &(optCosts[i]));
#ifdef MOSEK_GET_SOLUTION_STATISTICS
                MSK_getsolsta (task,MSK_SOL_ITR,&solsta);
                switch(solsta)
                {
                case MSK_SOL_STA_OPTIMAL:
                case MSK_SOL_STA_NEAR_OPTIMAL:
                    //MSK_getxx(task, MSK_SOL_ITR, xOpt.data());   /* Request the interior solution. */
                    break;
                case MSK_SOL_STA_DUAL_INFEAS_CER:
                case MSK_SOL_STA_PRIM_INFEAS_CER:
                case MSK_SOL_STA_NEAR_DUAL_INFEAS_CER:
                case MSK_SOL_STA_NEAR_PRIM_INFEAS_CER:
                    printf("Primal or dual infeasibility certificate found.\n"); break;
                case MSK_SOL_STA_UNKNOWN:
                    printf("The status of the solution could not be determined.\n"); break;
                default:
                    printf("Other solution status."); break;
                }
#endif
            }
            else
                printf("Error while optimizing.\n");
        }
        else
        {
            char symname[MSK_MAX_STR_LEN], desc[MSK_MAX_STR_LEN];
            MSK_getcodedesc (r,symname,desc);
            printf("Error while optimizing: %s - '%s'\n",symname,desc);
        }
	}
	t = clock() - t;

    avgTime = ((float)t)/(CLOCKS_PER_SEC*nTest);
    MSK_deletetask(&task);
    MSK_deleteenv(&env);

    return true;
}

testQPsolvers
=============

Tests of different QP solvers for application to real-time floating-base robot control.

## Goal
The goal is to find the fastest solver for this kind of application.
The optimization problems I am interested in are characterized by:
- small number of variable in the order of the number of joints of the robot (e.g. 20/40)
- small number of inequality constraints in the order of the number of variables (e.g. 20/40)
- subsequent resolution of similar problems (i.e. use of warm start is beneficial)
- hard constraints on computation time (typically around 1 ms)

## List of tested QP solvers
Currently the code tests the following solvers:
- Mosek
- OOQP (Object-Oriented QP)
- EigQuadProg
- QP Oases

Moreover I implemented two naive solvers for equality-constrained QP using the Eigen library.
This is to have a lower bound on the computation time, given that the resolution of a QP with inequality constraints using an active-set method boils down to the resolution of a sequence of equality-constrained QP.
The two naive solvers differ from each other for the way they solve the KKT system. In one case the KKT system is solved directly (i.e. decomposing the full KKT matrix), while in the other case the KKT system is solver by block elimination (i.e. by forming the Schur complement). In theory the two approaches should be computationally equivalent (however in this particolar implementation they are not).

## Results
Just for information I report here the results I measured on my machine (Ubuntu 12.04, 4 GB ram, 3.3 GHz quad-core cpu).

### 1000 QP, 30 variables, 24 inequality constraints
At each iteration the gradient is perturbed by a Gaussian random variable with variance 0.01, while the Hessian is perturbed by a Gaussian random variable with variance 0.1.
* TEST EIGEN EQUALITY FULL KKT, average time for QP: 0.03 ms
* TEST EIGEN EQUALITY ELIMINATION, average time for QP: 0.03 ms
* TEST MOSEK, average time for QP: 4.14 ms
* TEST OOQP Dense, average time for QP: 1.69 ms
* TEST EigQuadProg, average time for QP: 0.05 ms
* TEST QP OASES, average time for QP: 0.11 ms

Average number of active set reconstructions: 0.048
Average number of active constraints: 6.623

### 1000 QP, 30 variables, 44 inequality constraints
At each iteration the gradient is perturbed by a Gaussian random variable with variance 0.01, while the Hessian is perturbed by a Gaussian random variable with variance 0.1.
* TEST EIGEN EQUALITY FULL KKT: Average time for QP: 0.03 ms
* TEST EIGEN EQUALITY ELIMINATION: Average time for QP: 0.03 ms
* TEST MOSEK: Average time for QP: 5.89 ms
* TEST EigQuadProg: Average time for QP: 0.17 ms
* TEST QP OASES: Average time for QP: 0.25 ms

Average number of active set reconstructions: 0.061
Average number of active constraints: 11.044



Comments and Conclusions
===================
From the tests I ran, it seems that EigQuadProg is the fastest QP solver for dense problems with less than 50 variables and 50 inequality constraints. 
Of course these tests are not fair because most solvers (but not EigQuadProg) use BLAS for linear algebra, so changing my standard version of BLAS with an optimized version could affect the results.

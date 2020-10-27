# Distributed Matrix Computation

Performed distributed computing with the MPI- message passing interface. Implement fox algorithm to multiply two square matrices.

-------------------------------------------------------------------------------------------------------------------------------------

Source files:

mpi_matrix_mult.c :This program contains the parallel program to perform matrix multiplication using MPI and OpenMP.

Makefile: Contains code to build and run project.

mult.job: Slurm script to run the multiplication program

-------------------------------------------------------------------------------------------------------------------------------------

Running the code:

1. Check the arguments in Line 14 in the 'mult.job' scipt

    first argument: 'n' i.e. the size of the matrix (ranging from 4 to 14000)

    second argument: '0' for no verification, '1' for verification.

2. Run: mult.job
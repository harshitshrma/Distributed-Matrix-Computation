EXECS=mpi_matrix_mult
MPICC?=mpicc

all: ${EXECS}

mpi_matrix_mult: mpi_matrix_mult.c
	${MPICC} -O2 -fopenmp mpi_matrix_mult.c -o mpi_matrix_mult -lm

clean:
ifneq ("", "$(wildcard $(EXECS))")
	rm ${EXECS}
endif

zipcopy:
	tar cvzf lab5.tar.gz Makefile README.md *.pptx *.xlsx *.c *.h *.job

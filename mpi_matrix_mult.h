#ifndef __MAT_MULT_H
#define __MAT_MULT_H

#define FLOAT double
#define FLOAT_MPI MPI_DOUBLE
/* Type define structure of process grid (Cartesian)*/
typedef struct {
    int       p;             /* Total number of processes    */
    MPI_Comm  comm;          /* Communicator for entire grid */
    MPI_Comm  row_comm;      /* Communicator for my row      */
    MPI_Comm  col_comm;      /* Communicator for my col      */
    int       q;             /* Order of grid                */
    int       my_row;        /* My row number                */
    int       my_col;        /* My column number             */
    int       my_rank;       /* My rank in the grid comm     */
} grid_info_t;

/* Type define structure of local matrix */
#define MAX 14000*14000  // Maximum number of elements in array that store local matrix (2^16)
typedef struct {
    int     n_bar;
    #define Order(A) ((A)->n_bar)
    FLOAT  entries[MAX];
    #define Entry(A,i,j) (*(((A)->entries) + ((A)->n_bar)*(i) + (j)))
} local_matrix_t;

local_matrix_t *local_matrix_allocate(int n_bar);
void free_local_matrix(local_matrix_t** local_A);
void create_matrix(local_matrix_t* local_A, grid_info_t* grid, int n);
void save_matrix(local_matrix_t* local_A, grid_info_t* grid, int n, FLOAT *arr);
void set_to_zero(local_matrix_t* local_A);
void local_matrix_multiply(local_matrix_t* local_A,
    local_matrix_t* local_B, local_matrix_t* local_C);
void build_matrix_type(local_matrix_t* local_A);
MPI_Datatype local_matrix_mpi_t;
local_matrix_t *temp_mat;

void setup_grid(grid_info_t *grid);
void fox(int n, grid_info_t *grid, local_matrix_t *local_A, local_matrix_t
    *local_B, local_matrix_t *local_C);
bool verify_result(FLOAT *check_A, FLOAT *check_B, FLOAT *result, int n);
double random_real(double low, double high);
#endif

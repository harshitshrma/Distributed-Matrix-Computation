#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include "mpi_matrix_mult.h"

int main(int argc, char *argv[])
{
  int my_rank;
  grid_info_t grid;
  local_matrix_t *local_A;
  local_matrix_t *local_B;
  local_matrix_t *local_C;
  int n;
  int n_bar;
  int verify = 0;
  double timer_start;
  double timer_end;
  FLOAT *check_A = NULL;
  FLOAT *check_B = NULL;


  if( argc > 3 ) {
      printf("Too many arguments supplied.\n");
   } else if (argc < 3) {
      printf("Two arguments expected.\n");
   }

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  setup_grid(&grid);

  if (my_rank == 0) {
    n = atoi(argv[1]);
    printf("Order of matrices: %d\n", n);
    verify = atoi(argv[2]);
    if (verify) {
      check_A = malloc(n*n*sizeof(FLOAT));
      check_B = malloc(n*n*sizeof(FLOAT));
    }
  }
  local_A = local_matrix_allocate(n_bar);
  local_B = local_matrix_allocate(n_bar);
  temp_mat = local_matrix_allocate(n_bar);
  local_C = local_matrix_allocate(n_bar);
  FLOAT *result = malloc(n*n*sizeof(FLOAT));
  timer_start = MPI_Wtime();
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // local matrix's order
  n_bar = n/grid.q;

  
  Order(local_A) = n_bar;
  create_matrix(local_A, &grid, n);
  if (verify)
    save_matrix(local_A, &grid, n, check_A);

  
  Order(local_B) = n_bar;
  create_matrix(local_B, &grid, n);
  if (verify)
    save_matrix(local_B, &grid, n, check_B);

  // Buid local_A's MPI matrix data type
  build_matrix_type(local_A);
  
  Order(local_C) = n_bar;
  
  //fox algorithm
  MPI_Barrier(MPI_COMM_WORLD);
  
  fox(n, &grid, local_A, local_B, local_C);
  
  MPI_Barrier(MPI_COMM_WORLD);

  

  save_matrix(local_C, &grid, n, result);
  timer_end = MPI_Wtime();

  free_local_matrix(&local_A);
  free_local_matrix(&local_B);
  free_local_matrix(&local_C);

  //verification part
  if (my_rank == 0){
    printf("Matrix Multiplication Elapsed time: %e seconds\n", timer_end-timer_start);
    if (verify) {
    if (verify_result(check_A, check_B, result, n))
      printf("Resultant matrix is verified\n");
    else
      printf("Resultant matrix doesn't match\n");
    }
  }
  free(result);
  if (verify) {
    free(check_A);
    free(check_B);
  }
  MPI_Finalize();
}

void build_matrix_type(local_matrix_t *local_A)
{
  MPI_Datatype  temp_mpi_t;
  int           block_lengths[2];
  MPI_Aint      displacements[2];
  MPI_Datatype  typelist[2];
  MPI_Aint      start_address;
  MPI_Aint      address;
  MPI_Type_contiguous(Order(local_A)*Order(local_A), FLOAT_MPI, &temp_mpi_t);
  block_lengths[0] = block_lengths[1] = 1;
  typelist[0] = MPI_INT;
  typelist[1] = temp_mpi_t;
  MPI_Get_address(local_A, &start_address);
  MPI_Get_address(&(local_A->n_bar), &address);
  displacements[0] = address - start_address;
  MPI_Get_address(local_A->entries, &address);
  displacements[1] = address - start_address;
  MPI_Type_create_struct(2, block_lengths, displacements, typelist, &local_matrix_mpi_t);
  MPI_Type_commit(&local_matrix_mpi_t);
}

/* Create and distribute matrix:
 *     for each global row of the matrix,
 *         for each grid column
 *             create a block of n_bar floats on process 0
 *             and send them to the appropriate process.
 */
void create_matrix(local_matrix_t *local_A, grid_info_t *grid, int n)
{
  int mat_row, mat_col;
  int grid_row, grid_col;
  int dest;
  int coords[2];
  FLOAT *temp;
  MPI_Status status;

  if (grid->my_rank == 0) {
    temp = (FLOAT*) malloc(Order(local_A)*sizeof(FLOAT));
    for (mat_row = 0;  mat_row < n; mat_row++) {
      grid_row = mat_row/Order(local_A);
      coords[0] = grid_row;
      for (grid_col = 0; grid_col < grid->q; grid_col++) {
        coords[1] = grid_col;
        MPI_Cart_rank(grid->comm, coords, &dest);
        if (dest == 0) {
            for (mat_col = 0; mat_col < Order(local_A); mat_col++)
              *((local_A->entries)+mat_row*Order(local_A)+mat_col) =
                  random_real(1, 10);
        } else {
            for(mat_col = 0; mat_col < Order(local_A); mat_col++)
              *(temp + mat_col) = random_real(1, 10);
              MPI_Send(temp, Order(local_A), FLOAT_MPI, dest, 0, grid->comm);
            }
          }
      }
    free(temp);
  } else {
    for (mat_row = 0; mat_row < Order(local_A); mat_row++)
      MPI_Recv(&Entry(local_A, mat_row, 0), Order(local_A), FLOAT_MPI, 0, 0,
        grid->comm, &status);
  }
}

local_matrix_t *local_matrix_allocate(int local_order)
{
  local_matrix_t *temp;
  temp = (local_matrix_t*) malloc(sizeof(local_matrix_t));
  return temp;
}

void setup_grid(grid_info_t *grid)
{
  int old_rank;
  int dimensions[2];
  int wrap_around[2];
  int coordinates[2];
  int free_coords[2];

  MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));
  MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);

  grid->q = (int) sqrt((FLOAT) grid->p);
  dimensions[0] = dimensions[1] = grid->q;

  wrap_around[0] = wrap_around[1] = 1;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, wrap_around, 1, &(grid->comm));
  MPI_Comm_rank(grid->comm, &(grid->my_rank));
  MPI_Cart_coords(grid->comm, grid->my_rank, 2, coordinates);
  grid->my_row = coordinates[0];
  grid->my_col = coordinates[1];

  /* Set up row communicators */
  free_coords[0] = 0;
  free_coords[1] = 1;
  MPI_Cart_sub(grid->comm, free_coords, &(grid->row_comm));

  /* Set up column communicators */
  free_coords[0] = 1;
  free_coords[1] = 0;
  MPI_Cart_sub(grid->comm, free_coords, &(grid->col_comm));
}

void fox(int n, grid_info_t *grid, local_matrix_t*  local_A, local_matrix_t*  local_B,
  local_matrix_t*  local_C)
{
  local_matrix_t*  temp_A;
  int              stage;
  int              bcast_root;
  int              n_bar;
  int              source;
  int              dest;
  MPI_Status       status;
  n_bar = n/grid->q;
  set_to_zero(local_C);

  /* Calculate addresses for row circular shift of B */
  source = (grid->my_row + 1) % grid->q;
  dest = (grid->my_row + grid->q - 1) % grid->q;

  /* Set aside storage for the broadcast block of A */
  temp_A = local_matrix_allocate(n_bar);

  for (stage = 0; stage < grid->q; stage++) {
    bcast_root = (grid->my_row + stage) % grid->q;
      if (bcast_root == grid->my_col) {
         MPI_Bcast(local_A, 1, local_matrix_mpi_t, bcast_root, grid->row_comm);
          local_matrix_multiply(local_A, local_B, local_C);
      } else {
            MPI_Bcast(temp_A, 1, local_matrix_mpi_t, bcast_root, grid->row_comm);
            local_matrix_multiply(temp_A, local_B, local_C);
      }
      MPI_Sendrecv_replace(local_B, 1, local_matrix_mpi_t, dest, 0, source, 0,
        grid->col_comm, &status);
    }
}

void set_to_zero(local_matrix_t *local_A)
{
  int i, j;
  for (i = 0; i < Order(local_A); i++)
    for (j = 0; j < Order(local_A); j++)
      Entry(local_A,i,j) = 0.0E0;
}

/* local matrix multiplication function
*  with OpenMP Thread Acceleration
*/
void local_matrix_multiply(local_matrix_t*  local_A, local_matrix_t*  local_B,
    local_matrix_t*  local_C) {
    int i, j, k;

    #pragma omp parallel for private(i, j, k) shared(local_A, local_B, local_C)
    for (i = 0; i < Order(local_A); i++) {
      for (j = 0; j < Order(local_A); j++)
        for (k = 0; k < Order(local_B); k++)
          Entry(local_C,i,j) = Entry(local_C,i,j) + Entry(local_A,i,k)*Entry(local_B,k,j);
    }
}

/* Recive and save Matrix:
 *     for each global row of the matrix,
 *         for each grid column
 *             send n_bar floats to process 0 from each other process
 *             receive a block of n_bar floats on process 0 from other processes and print them
 */
void save_matrix(local_matrix_t*  local_A, grid_info_t *grid, int n, FLOAT* arr) {
  int mat_row, mat_col;
  int grid_row, grid_col;
  int source;
  int coords[2];
  FLOAT* temp;
  MPI_Status status;

  if (grid->my_rank == 0) {
    temp = (FLOAT*) malloc(Order(local_A)*sizeof(FLOAT));
      for (mat_row = 0;  mat_row < n; mat_row++) {
        grid_row = mat_row/Order(local_A);
        coords[0] = grid_row;
        for (grid_col = 0; grid_col < grid->q; grid_col++) {
            coords[1] = grid_col;
            MPI_Cart_rank(grid->comm, coords, &source);
            if (source == 0) {
              for(mat_col = 0; mat_col < Order(local_A); mat_col++)
                  *((arr+mat_row*n) + mat_col) = Entry(local_A, mat_row, mat_col);
                } else {
                    MPI_Recv(temp, Order(local_A), FLOAT_MPI, source, 0, grid->comm, &status);
                    for(mat_col = 0; mat_col < Order(local_A); mat_col++)
                      *((arr+mat_row*n) + mat_col) = temp[mat_col];
                }
            }
        }
    free(temp);
  } else {
      for (mat_row = 0; mat_row < Order(local_A); mat_row++)
        MPI_Send(&Entry(local_A, mat_row, 0), Order(local_A), FLOAT_MPI, 0, 0, grid->comm);
    }

}

void free_local_matrix(local_matrix_t **local_A_ptr)
{
  free(*local_A_ptr);
}

bool verify_result(FLOAT *check_A, FLOAT *check_B, FLOAT *result, int n) {
    int i, j, k;
    bool flag = true;

    #pragma omp parallel for
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        FLOAT tmp = 0;
        for (k = 0; k < n; k++) {
          tmp += check_A[i*n + k]*check_B[k*n + j];
        }
        if (result[i*n + j] != tmp)
         flag = false;
      }
    }
    return flag;
}

double random_real(double low, double high)
{
  double d;
  d = (double) rand() / ((double) RAND_MAX + 1);
  return (low + d * (high - low));
}

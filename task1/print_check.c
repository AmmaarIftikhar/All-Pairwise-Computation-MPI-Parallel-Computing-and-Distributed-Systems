#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "pairwise_comp.h"
#include "distributed_comp.h"
#define ALIGNMENT 64
#include <time.h>
#include <sys/time.h>
#include <string.h>

int main (int argc, char** argv) {

  // VAR DECLARATION & INIT
  void** matrix_local, **A;
  int rank; // create a set rank function
  int d1, d2, ty;
  int num_proc, size;
  
  // INITIALIZE MPI 
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  srand(1558);
  arg_check(argc, argv, rank, &d1, &d2, &ty);

  set_rank(rank);
  set_dims(d1, d2);
  set_ty_num(ty, num_proc);
  
  
  if (rank == 0) {
    // initial array allocation
    if (ty == 1)
      A = (void**) allocate_array_db(d1, d2, 1);
    else
      A = (void**) allocate_array_fl(d1, d2, 1);
  }
  
  set_distparams();
  size = get_size();
  set_unroll(0);
  matrix_local = (void**) malloc(size*sizeof(void*));
  distributed(matrix_local, A);
  MPI_Barrier(MPI_COMM_WORLD);
  
  set_unroll(0);
  
  // printing and checking computation  
  if (rank == 0) {
    int check = 0;
    void* ser_results;

    ser_results = run_serial(A);

    print_serial(ser_results, d2);
    print_2dresults();

    check = compare(ser_results);

    if (check > 0)
      printf("failed equality %d\n", check);
    else
      printf("success equality, %d \n", check);
    
    free(ser_results);
  }
  
  if (rank == 0)
    delete_matrix(A, d2);

  free_local(matrix_local);
  
  // end
  MPI_Finalize();
  
  return 0;
}

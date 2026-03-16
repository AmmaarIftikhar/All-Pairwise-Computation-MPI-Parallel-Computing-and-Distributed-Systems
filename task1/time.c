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
  struct timeval tval_before, tval_after, tval_result;
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

  gettimeofday(&tval_before, NULL);
  
  if (num_proc > 1) {
    set_distparams();
    size = get_size();
    matrix_local = (void**) malloc(size*sizeof(void*));
    distributed(matrix_local, A);
    MPI_Barrier(MPI_COMM_WORLD);
  } else {
    void* ser_results;
    ser_results = run_serial(A);
    free(ser_results);
  }

  gettimeofday(&tval_after, NULL);
  timersub(&tval_after, &tval_before, &tval_result);
    
  // printing and checking computation  
  if (rank == 0) {
    printf("time = %f sec\n", (double)(tval_result.tv_sec*1000000 + tval_result.tv_usec)/1e6);
  }
  
  if (rank == 0)
    delete_matrix(A, d2);

  free_local(matrix_local);
  
  // end
  MPI_Finalize();
  
  return 0;
}

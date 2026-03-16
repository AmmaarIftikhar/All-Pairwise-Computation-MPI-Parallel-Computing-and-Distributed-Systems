#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "pairwise_comp.h"
#define ALIGNMENT 64
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include "distributed_comp.h"
#include <math.h>
int num_proc;
int total_transfers, buffer_size;
int size, results_size;

// let result be a 1D array
void *results;
void** final_results;
int d1, d2;
int rank, ty;
MPI_Datatype dtype;   // will hold MPI_FLOAT or MPI_DOUBLE
size_t esize;

// add more checks
int arg_check(int argc, char** argv, int rank, int* d1, int* d2, int* ty) {
  char* typ;

  if (argc == 4) {
    *d1 = atoi(argv[1]);
    *d2 = atoi(argv[2]);
    typ = argv[3];

    if (*d1 < 1 || *d2 < 1) {

      if (rank == 0)
	fprintf(stderr, "Error: failed dimension value\n");

      MPI_Finalize();
      exit(1);
    }
  } else {
    
    if (rank == 0)
      fprintf(stderr, "Error: wrong number of dimensions\n");

    MPI_Finalize();
    exit(1);
  }

  if (strcmp(typ, "double") == 0) {
    *ty = 1;
  } else if (strcmp(typ, "float") == 0) {
    *ty = 0;
  } else {
    if (rank == 0)
      fprintf(stderr, "Error: 3rd argument must be exactly \"float\" or \"double\"\n");
    MPI_Finalize();
    exit(1);
  }

  set_type(*ty);

  return 0;
}


// changed tag to row index
int send_init(void** A) {

  int tag = size;
  int sz;
  
  for (int i = 1; i < num_proc; i++) {
    sz = d2 / num_proc;
    
    if (i < (d2 % num_proc))
      sz = sz + 1;

    for (int j = 0; j < sz; j++) {
      MPI_Send(A[tag], d1, dtype, i, tag, MPI_COMM_WORLD); // send (size*i + j)th block to process i
      tag++;
    }    
  }

  return 0;
}

// changed tag to row index
int recv_init(void** A) {

  int tag = (d2 / num_proc) * rank;

  if (rank >= (d2 % num_proc))
    tag = tag + (d2 % num_proc);
  else if (rank < (d2 % num_proc))
    tag = tag + rank;
  
  for (int j = 0; j < size; j++) {
    MPI_Recv(A[j], d1, dtype, 0, tag + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return 0;
}

int communicate(void* outdata, void* indata, int left_neigbor, int right_neighbor, int index) {
  int in_tag, out_tag;
  // send first, receive later

  if (rank % 2 == 0) {
    MPI_Send(outdata, d1, dtype, right_neighbor, index, MPI_COMM_WORLD);
    MPI_Recv(indata, d1, dtype, left_neigbor, index, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(indata, d1, dtype, left_neigbor, index, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(outdata, d1, dtype, right_neighbor, index, MPI_COMM_WORLD);
  }

  return 0;
}

int communicate_compute(void** data) {
  
  int left_neigbor  = (rank-1) >= 0? (rank-1): num_proc-1;
  int right_neighbor = (rank+1) % num_proc;
  int result_index = 0, msg_index = 0;
  int index2 = 0;

  // only send its 'size' number of msgs
  // needs to send the ones it receives -- hwo many will it receive.  ??
  // estimate is floor of M/2
  
  void* cur_data, *msg;
  void** transfer_buffer;  

  transfer_buffer = malloc(sizeof(void*)*(size+1));
  
  for (int i = 0; i <= size; i++)
    posix_memalign(transfer_buffer + i, ALIGNMENT, d1*esize);

  // replace size with total columns / 2
  serial_pairwise_comp(data, results, d1, size);  
  result_index += (size*(size+1))/2;
  
  int temp = total_transfers;
  
  for (int i = 0; i < temp; i++) {
    msg = transfer_buffer[msg_index];
    
    msg_index = (msg_index + 1) % (size+1);
    cur_data =  (i < size)? data[size - i - 1]: transfer_buffer[index2];
    index2 = i < size? index2: (index2 + 1) % (size + 1);
    
    // get and send the data          // compute
    communicate(cur_data, msg, left_neigbor, right_neighbor, i);
    if (ty == 1) {
      for (int j = 0; j < size && i < total_transfers; j++) {
	col_mult((double*)results+result_index, data[j], msg, 0, d1);
	result_index++;
      }
    } else {
      for (int j = 0; j < size && i < total_transfers; j++) {
        col_mult((float*)results+result_index, data[j], msg, 0, d1);
        result_index++;
      }
    }
  }

  
  for (int i = 0; i <= size; i++)
    free(transfer_buffer[i]);

  free(transfer_buffer);

  return 0;
}

// -mod- changes to sz
//       changes to pos
void results_proc() {
  void *temp_data;
  int pos[3] = {0, 0, 0};

  for (int i = 0; i < num_proc; i++) {

    int sz = d2 / num_proc;
    if (i < (d2 % num_proc))
      sz = sz + 1;

    // changes size to sz
    int transfer_est = (sz*(sz + 1))/2 + (((d2/2 + (d2 % num_proc) + 2)*sz)); // check estimation correctness
    
    // for msg indexing and processing
    int msg_count = 0, tag = 0;
    int x  = 0, y = 0;

    // recv msg indexing and tracking
    pos[1] = 0, pos[2] = 0;

    // locates the first left neighbor
    int src = (pos[0] - 1 + d2) % d2;
    
    while (transfer_est > pos[1]) {

      if (msg_count == 0) {
	msg_count = buffer_size <= (transfer_est - (tag*buffer_size))? buffer_size: (transfer_est - (tag*buffer_size));

	if (i == 1 && tag == 0)
	  temp_data = malloc(esize*buffer_size);

	if (i != 0)
	  MPI_Recv(temp_data, msg_count, dtype, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	else {
	  temp_data = results;
	  msg_count = results_size;
	}

	pos[2] = 0;
	tag++;
      }

      if (pos[1] < (sz*(sz+1))/2) {
	if (ty == 1)
	  ((double**)final_results)[pos[0] + x][pos[0] + y] = ((double*)temp_data)[pos[2]];
	else
	  ((float**)final_results)[pos[0] + x][pos[0] + y] = ((float*)temp_data)[pos[2]];

	int x_prev = x;
	
	x = (((y + 1) / (x+1)) + x) % sz;
	y = (y + 1) % (x_prev+1);

      } else {
	if (ty == 1) {
	  if (src <= pos[0]) // if < pos0 then less than all pos0 + x and vice versa
	    ((double**)final_results)[pos[0] + y][src] = ((double*)temp_data)[pos[2]];
	  else 
	    ((double**)final_results)[src][pos[0] + y] = ((double*)temp_data)[pos[2]];
	} else {
	  if (src <= pos[0]) // if < pos0 then less than all pos0 + x and vice versa                                                                                                                                                         
            ((float**)final_results)[pos[0] + y][src] = ((float*)temp_data)[pos[2]];
          else
            ((float**)final_results)[src][pos[0] + y] = ((float*)temp_data)[pos[2]];
	}
	  
	src = (src - ((y+1) / sz) + d2) % d2;
	y = (y+1) % sz;
	
      }
      
      pos[2]++;
      pos[1]++;
      msg_count--;
    }
    
    pos[0] += sz;
  }

  if (num_proc > 1)
    free(temp_data);
}


void send_final() {
  int tag = 0, pos = 0;
  int msg_count = buffer_size <= total_transfers*size? buffer_size: total_transfers*size;
  int transfer_est = (size*(size + 1))/2 + (((total_transfers)*size));
  
  while (msg_count > 0) {
    if (ty == 1)
      MPI_Send((double*)results + pos, msg_count, dtype, 0, tag, MPI_COMM_WORLD);
    else
      MPI_Send((float*)results + pos, msg_count, dtype, 0, tag, MPI_COMM_WORLD);
    
    pos += msg_count;
    tag++;
    
    msg_count = buffer_size <= ((transfer_est) - (tag*buffer_size))? buffer_size:
      ((transfer_est) - (tag*buffer_size));
  }
}


void print_serial(void* serial_results, int d2) {
  int pos = 0;
  
  for (int i = 0; i < d2; i++) {
    printf("%d: ", i);

    for (int j = 0; j <= i; j++) {
      if (ty == 1)
	printf("%f ", ((double*)serial_results)[pos]);
      else
	printf("%f ", ((float*)serial_results)[pos]);
      pos++;
    }

    printf("\n");
  }
}

void print_2dresults() {
  void** fresults = final_results;
  
  for (int i = 0; i < d2; i++) {
    printf("%d: ", i);
      
    for (int j = 0; j <= i; j++) {
      if (ty == 1)
	printf("%f ", ((double**)fresults)[i][j]);
      else
 	printf("%f ", ((float**)fresults)[i][j]);
    }
    
    printf("\n");
  }

}


int compare(void* serial_results) {
  int pos = 0;
  int check = 0;
  void** fresults = final_results;
  
  for (int i = 0; i < d2; i++) {

    for (int j = 0; j <= i; j++) {
      if (ty == 1) {
	if (((double**)fresults)[i][j] - ((double*)serial_results)[pos] > 0.0001 ||
	    ((double**)fresults)[i][j] - ((double*)serial_results)[pos] < -0.0001)
	  check++;
      } else {
	if (fabsf(((float**)fresults)[i][j] - ((float*)serial_results)[pos]) > 1e-3f)
          check++;
      }
      pos++;
    }
  }

  return check;
}

void set_distparams() {
  
  // set parameters
  size = d2 / num_proc;  
  buffer_size = d1>d2? d1: d2;
  buffer_size = buffer_size > d2 + (d1 / num_proc)? buffer_size:
    d2 + (d1 / num_proc);
  
  // set the total number of transfers
  if (rank < (d2 % num_proc))
      size = size + 1;
  
  total_transfers = (d2/2) + (d2 % num_proc) + 2;
}

void distributed(void** matrix_local, void** A) {

  results_size = ((size*(size+1))/2) + ((total_transfers)*size);
  
  posix_memalign(&results, ALIGNMENT, results_size * esize);

  if (rank > 0) {

    // local matrix allocation   
    for (int i = 0; i < size; i++)
      posix_memalign(matrix_local + i, ALIGNMENT, d1*esize);

    // recv data from 0
    recv_init(matrix_local);
  } else {
    
    final_results = malloc( d2 * sizeof(void*));
    
    for (int i = 0; i < size; i++)
      matrix_local[i] = A[i];

    for (int i = 0; i < d2; i++)
      final_results[i] = malloc((i+1)*esize);

    send_init(A);    
  }
    
  // Stores time seconds  
  set_unroll(1);

  
  MPI_Barrier(MPI_COMM_WORLD);
  communicate_compute(matrix_local);
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0)
     results_proc();
  else
    send_final();
  
  MPI_Barrier(MPI_COMM_WORLD);
  
}


void* run_serial(void** A) {
  void* ser_results;
  
  posix_memalign(&ser_results, ALIGNMENT, (d2*(d2+1)/2) * esize);
  serial_pairwise_comp(A, ser_results, d1, d2);

  return ser_results;
}

void free_local(void** matrix_local) {
  
  if (rank != 0) {
    for (int i = 0; i < size; i++)
      free(matrix_local[i]); 
  }
  
  if (num_proc > 1) {
    free(results);
    free(matrix_local);
  }

  
  if (rank == 0) {    
    if (num_proc > 1) {
      for (int i = 0; i < d2; i++)
	free(final_results[i]);
      
      free(final_results);
    }
  }
  
}

void set_dims(int dd1, int dd2) {
  d1 = dd1;
  d2 = dd2;
}

void set_rank(int r) {
  rank = r;
}

void set_ty_num(int t, int n) {
  ty = t;
  
  if (ty == 1) {
    dtype = MPI_DOUBLE;
    esize = sizeof(double);
  } else {
    dtype = MPI_FLOAT;
    esize = sizeof(float);
  }

  num_proc = n;
}

int get_size() {
  return size;
}

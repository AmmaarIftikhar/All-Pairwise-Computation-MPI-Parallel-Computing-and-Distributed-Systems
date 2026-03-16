#include <stdio.h>
#include <stdlib.h>
#include "pairwise_comp.h"

#define ALIGNMENT 64

int type = 0;
int unroll = 0;

/*
  instead of creating a NxM matrix,
  create a MxN matrix for spatial locality,
  or maybe even create a N*M 1D matrix  -- better for communication
 */
double** allocate_array_db(int d1, int d2, int mode) {
  double** arr =  (double**) malloc(sizeof(double*) * d2);

    for (int i = 0; i < d2; i++) {

      posix_memalign((void**)&arr[i], ALIGNMENT, d1*sizeof(double) );

      for (int k =0; k < d1; k++)
        arr[i][k] = mode * (double) rand() / RAND_MAX;
    }

  return arr;
}

float** allocate_array_fl(int d1, int d2, int mode) {
  float** arr =  (float**) malloc(sizeof(float*) * d2);

  for (int i = 0; i < d2; i++) {
    posix_memalign((void**)&arr[i], ALIGNMENT, d1*sizeof(float) );
      
    for (int k = 0; k < d1; k++)
        arr[i][k] = mode * (float) rand() / RAND_MAX;

    }

  return arr;
}

void col_mult(void* c, void* a, void* b, int s1, int e1) {

  if (type == 0)
    *(float*)c = 0;
  else
    *(double*)c = 0;
  
  for (int i = s1; i < e1; i ++) {

    if (type == 0)
      *(float*)c += ((float*)a)[i] * ((float*)b)[i];
    else 
      *(double*)c += ((double*)a)[i] * ((double*)b)[i];
    
  }
}

void col_mult_unroll(void* c, void* a, void* b, int s1, int e1) {

  int len = e1 - s1;
  int i = s1;

  if (type == 0) {
    float* af = (float*)a;
    float* bf = (float*)b;
    float  sum = 0.0f;


    int limit = s1 + (len / 4) * 4;
    for (i = s1; i < limit; i += 4) {
      sum += af[i]   * bf[i]
	+ af[i+1] * bf[i+1]
	+ af[i+2] * bf[i+2]
	+ af[i+3] * bf[i+3];
    }

    for (; i < e1; i++) {
      sum += af[i] * bf[i];
    }

    *(float*)c = sum;
  }
  else {
    double* ad = (double*)a;
    double* bd = (double*)b;
    double  sum = 0.0;

    int limit = s1 + (len / 4) * 4;
    for (i = s1; i < limit; i += 4) {
      sum += ad[i]   * bd[i]
	+ ad[i+1] * bd[i+1]
	+ ad[i+2] * bd[i+2]
	+ ad[i+3] * bd[i+3];
    }

    for (; i < e1; i++) {
      sum += ad[i] * bd[i];
    }

    *(double*)c = sum;
  }
}

int serial_pairwise_comp(void** A, void* C, int d1, int d2) {
  int pos = 0;
  void (*colmul)(void* c, void* a, void* b, int s1, int e1);
  
  if (unroll == 1)
    colmul = &col_mult_unroll;
  else
    colmul = &col_mult;
  
  for (int i = 0; i < d2; i++) {
    
    for (int j = 0; j <= i; j++) {
      
      if (type == 1)
	colmul(((double*)C)+pos, A[i], A[j], 0, d1);
      else
	colmul(((float*)C)+pos, A[i], A[j], 0, d1);

      pos++;
    }  
  }
  
  return 0;
}

void set_type(int t) {
  type = t;
}

void set_unroll(int u) {
  if (u == 0)
    unroll = 0;
  else
    unroll = 1; 
}

void delete_matrix(void** arr, int d2) {

  if (type == 0) {
    for (int i = 0; i < d2; i++)
      free((float*)arr[i]);

    free((float**)arr);
  } else {
    for (int i = 0; i < d2; i++)
      free((double*)arr[i]);

    free((double**)arr);
  }    
}

/*
  distribute using process 0

  communication between i & i - 1 process
  - linear communication
  i receives from i-1,
  i sends to i+1

  if even:
     send
     recv
  else
     recv
     send

  
*/

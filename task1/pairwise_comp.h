#ifndef PAIRWISE_COMP_H
#define PAIRWISE_COMP_H
#include <stdio.h>
#include <stdlib.h>

double** allocate_array_db(int d1, int d2, int mode);
float** allocate_array_fl(int d1, int d2, int mode);
void col_mult(void* c, void* a, void* b, int s1, int e1);
void col_mult_unroll(void* c, void* a, void* b, int s1, int e1);
int serial_pairwise_comp(void** A, void* C, int d1, int d2);
void set_type(int t);
void set_unroll(int u);
void delete_matrix(void** arr, int d1);
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

#endif

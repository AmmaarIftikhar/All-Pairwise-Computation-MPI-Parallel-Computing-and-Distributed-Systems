#ifndef DISTRIBUTED_COMP_H
#define DISTRIBUTED_COMP_H

// add more checks
int arg_check(int argc, char** argv, int rank, int* d1, int* d2, int* ty);
void print_serial(void* serial_results, int d2);
void print_2dresults();
int compare(void* serial_results);
void set_distparams();
void distributed(void** matrix_local, void** A);


void* run_serial(void** A);
void free_local(void** matrix_local);
void set_dims(int dd1, int dd2);
void set_rank(int r);
int get_size();
void set_ty_num(int t, int n);

#endif

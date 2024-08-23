#include "mpi.h"
#include <stdio.h>
#include <cstdlib>

#define BYTES sizeof(int)

void vprintf(int *v, int size, int pid);
void initialize_row(int *row, int idx, int n);
int** initialize_all_rows(int start, int end, int n, int pid);

int main(int argc, char**argv){
    const int n = std::atoi(argv[1]);
    int pid, tasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    
    int tag = 0;
    int start = (n/tasks)*pid;
    int end = (n/tasks)*(1+pid);
    int** rows = initialize_all_rows(start, end, n, pid);
    if(pid == 0){
        printf("Matrix size: %d x %d\n", n, n);
        printf("---------------------------------------------------------------------------------------\n");
        for(int i=start; i < end; ++i){
            vprintf(rows[i], n, pid);
        }
        
        for(int i=1; i < tasks; ++i){
            int* pids_rows = (int*) malloc(n*BYTES);
            for(int j=start; j < end; ++j){
                MPI_Recv(pids_rows, n, MPI_INT, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                vprintf(pids_rows, n, i);
            } 
        }
        printf("---------------------------------------------------------------------------------------\n");
    }else{
        for(int i=start; i < end; ++i){
            MPI_Send(rows[i], n, MPI_INT, 0, tag, MPI_COMM_WORLD);
        }    
    }
    
    MPI_Finalize();  
    return EXIT_SUCCESS;
}

void vprintf(int *v, int size, int pid){
    for(int i=0; i < size; ++i){
        printf("%d\t", v[i]);
    }
    printf("\tProceso%d\n",pid);
}
void initialize_row(int *row, int idx, int n){
    for(int i=0; i < n; ++i){
        if(i != idx){
            row[i] = 0;
        }else{
            row[i] = 1;
        }
    }
}
int** initialize_all_rows(int start, int end, int n, int pid){
    int nrows = end-start;
    int** rows = (int**) malloc(n*BYTES*nrows);
    for(int i=start; i < end; ++i){
        int *irow = (int*) malloc(n*BYTES);
        initialize_row(irow, i, n);
        rows[i] = irow;
    }
    return rows;
}

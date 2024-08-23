#include "mpi.h"
#include <stdio.h>
#include <cstdlib>

#define BYTES sizeof(int)

void vprintf(int *v, int size);
void initialize_row(int *row, int idx, int n);
int* initialize_all_rows(int start, int end, int n, int pid);

int main(int argc, char**argv){
    //Amount of params is different to expected amount of params
    if(argc != 2) return EXIT_FAILURE;
    const int n = std::atoi(argv[1]);
    int pid, int tasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    //Amount of tasks is a N divisor?
    if(n/tasks != 0) return EXIT_FAILURE;
    std::printf("Matrix size: %d x %d\n\n", n, n);
    int start = (n/tasks)*pid;
    int end = (n/tasks)*(1+pid);
    int* rows = initialize_all_rows(start, end, n, pid);
    if(pid == 0){
        //TODO: Get rows from each pid 
        //TODO: Print each row
        for(int i=start; i < end; ++i){
            printf("Printing from pid %d\n", pid);
            vprintf(rows[i], n);
        }
    }
    }else{
        //TODO: Send rows to pid 0
        for(int i=start; i < end; ++i){
            printf("Printing from pid %d\n", pid);
            vprintf(rows[i], n);
        }
    }
    
    MPI_Finalize();  
    return EXIT_SUCCESS;
}

void vprintf(int *v, int size){
    for(int i=0; i < size; ++i){
        printf("%d\t", v[i]);
    }
    printf("\n");
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
int* initialize_all_rows(int start, int end, int n, int pid){
    int nrows = end-start;
    int* rows[nrows];
    for(int i=start; i < end; ++i){
        int *irow = (int*) malloc(n*SIZE);
        initialize_row(irow, i, n);
        rows[i] = irow;
    }
    return rows;
}
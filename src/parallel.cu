// Yuxuan, 27 June
// Parallel (CUDA) version of Hines algorthm.


#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cuda.h>

__global__ void HinesAlgo (
    double *u, double *l, double *d,
    double *rhs, int *p, int N
) {
    int i;
    double factor;
    int offset = blockIdx.x * N;

    for (i=N-1;i>=1;--i) {
        factor = u[i+offset] / d[i+offset];
        d[p[i]+offset] -= factor * l[i+offset];
        rhs[p[i]+offset] -= factor * rhs[i+offset];
    }

    rhs[0+offset] /= d[0+offset];

    for (i=1;i<=N-1;++i) {
        rhs[i+offset] -= l[i+offset] * rhs[p[i]+offset];
        rhs[i+offset] /= d[i+offset];
    }
}

// the main function receives 3 parameters: input path, output path and the number of repeated runs
// the number of repeated runs is required to be a multiple of 32
int main (int argc, char * argv[]) {
    FILE *fp;
    clock_t time;
    int runNum;
    cudaDeviceProp devProp;
    int blockNum, blockSize;

    // Host data
    int *id; double *u; double *l;
    double *d; double *rhs; int *p;

    // Device data
    int *id1; double *u1; double *l1;
    double *d1; double *rhs1; int *p1;

    int N;

    // read data
    fp = fopen(argv[1], "r");

    fscanf(fp, "%d", &N);

    id = new int [N]; u = new double [N]; l = new double [N];
    d = new double [N]; rhs = new double [N]; p = new int [N];

    for (int i=0;i<N;++i) {
        fscanf(
            fp, "%d %lf %lf %lf %lf %d",
            &id[i], &u[i], &l[i], &rhs[i], &d[i], &p[i]);
    }

    fclose(fp);

    runNum = atoi(argv[3]);

    // choose grid dim and block dim by number of SMs and number of runs
    blockNum = -1;

    for (blockSize=256;blockSize>=32;blockSize>>=1) {
        if (runNum%blockSize==0) {
            blockNum = runNum / blockSize;
            if (blockNum>=devProp.multiProcessorCount) {
                break;
            }
        }
    }

    if (blockNum==-1) {
        printf("Number of runs is not a multiple of 32.");
        return -1;
    }

    // allocate space for device data
    cudaMalloc(reinterpret_cast<void **>(&id1), N*sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&u1), runNum*N*sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&l1), runNum*N*sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&d1), runNum*N*sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&rhs1), runNum*N*sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&p1), N*sizeof(int));

    time = clock(); // include time for device memory copy so that comparison to serial code is fair

    // copy host data to device data
    cudaMemcpy(id1, id, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p1, p, N*sizeof(int), cudaMemcpyHostToDevice);

    for (int i=0;i<runNum;++i) {   
        cudaMemcpy(u1+i*N, u, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(l1+i*N, l, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d1+i*N, d, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(rhs1+i*N, rhs, N*sizeof(double), cudaMemcpyHostToDevice);
    }

    HinesAlgo <<<blockNum, blockSize>>> (u1, l1, d1, rhs1, p1, N);

    cudaDeviceSynchronize();

    // copy result back to host data
    cudaMemcpy(u, u1, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(l, l1, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(d, d1, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhs, rhs1, N*sizeof(double), cudaMemcpyDeviceToHost);

    time = clock() - time;

    printf("Parallel time cost of %d runs: %.2f seconds.\n", runNum, static_cast<double>(time)/CLOCKS_PER_SEC);

    // write result
    fp = fopen(argv[2], "w+");

    for (int i=0;i<N;++i) {
        fprintf(
            fp, "%d %lf %lf %lf %lf\n",
            id[i], u[i], l[i], rhs[i], d[i]);
    }

    fclose(fp);

    delete[] id; delete[] u; delete[] l;
    delete[] d; delete[] rhs; delete[] p;
}
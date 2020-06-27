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

    for (i=N-1+offset;i>=1+offset;--i) {
        factor = u[i] / d[i];
        d[p[i]+offset] -= factor * l[i];
        rhs[p[i]+offset] -= factor * rhs[i];
    }

    rhs[0+offset] /= d[0+offset];

    for (i=1+offset;i<=N-1+offset;++i) {
        rhs[i] -= l[i] * rhs[p[i]+offset];
        rhs[i] /= d[i];
    }
}

int main (int argc, char * argv[]) {
    FILE *fp;
    clock_t time;
    int repeatNum;

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

    if (argc==4) {
        repeatNum = atoi(argv[3]);
    } else {
        repeatNum = 1;
    }

    // allocate space for device data
    cudaMalloc(reinterpret_cast<void **>(&id1), repeatNum*N*sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&u1), repeatNum*N*sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&l1), repeatNum*N*sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&d1), repeatNum*N*sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&rhs1), repeatNum*N*sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&p1), repeatNum*N*sizeof(int));

    time = clock(); // include time for device memory copy so that comparison to serial code is fair

    // copy host data to device data
    for (int i=0;i<repeatNum;++i) {
        cudaMemcpy(id1+i*N, id, N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(u1+i*N, u, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(l1+i*N, l, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d1+i*N, d, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(rhs1+i*N, rhs, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(p1+i*N, p, N*sizeof(int), cudaMemcpyHostToDevice);
    }

    dim3 dimGrid (repeatNum, 1); // each copy of Hines system, one block
    dim3 dimBlock(1, 1, 1); // each block, one thread

    HinesAlgo <<<dimGrid, dimBlock>>> (u1, l1, d1, rhs1, p1, N);

    cudaDeviceSynchronize();

    // copy result back to host data
    cudaMemcpy(id, id1, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(u, u1, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(l, l1, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(d, d1, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhs, rhs1, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, p1, N*sizeof(int), cudaMemcpyDeviceToHost);

    time = clock() - time;

    printf("Serial time cost of %d runs: %.2f seconds.\n", repeatNum, static_cast<double>(time)/CLOCKS_PER_SEC);

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

    cudaFree(id1); cudaFree(u1); cudaFree(l1);
    cudaFree(d1); cudaFree(rhs1); cudaFree(p1);
}
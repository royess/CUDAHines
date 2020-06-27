#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include<cstring>

void HinesAlgo (
    double *u, double *l, double *d,
    double *rhs, int *p, int N
) {
    int i;
    double factor;

    for (i=N-1;i>=1;--i) {
        factor = u[i] / d[i];
        d[p[i]] -= factor * l[i];
        rhs[p[i]] -= factor * rhs[i];
    }

    rhs[0] /= d[0];

    for (i=1;i<=N-1;++i) {
        rhs[i] -= l[i] * rhs[p[i]];
        rhs[i] /= d[i];
    }
}

int main(int argc, char * argv[]) {
    FILE *fp;
    clock_t time;
    int repeatNum;

    int *id; double *u; double *l;
    double *d; double *rhs; int *p;

    int *id1; double *u1; double *l1;
    double *d1; double *rhs1; int *p1;

    int N;
    
    fp = fopen(argv[1], "r");

    fscanf(fp, "%d", &N);

    id = new int [N]; u = new double [N]; l = new double [N];
    d = new double [N]; rhs = new double [N]; p = new int [N];

    id1 = new int [N]; u1 = new double [N]; l1 = new double [N];
    d1 = new double [N]; rhs1 = new double [N]; p1 = new int [N];

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

    time = clock();

    for (int i=0;i<repeatNum;++i) {
        memcpy(u1, u, N*sizeof(double));
        memcpy(l1, l, N*sizeof(double));
        memcpy(d1, d, N*sizeof(double));
        memcpy(rhs1, rhs, N*sizeof(double));
        memcpy(p1, p, N*sizeof(int));

        HinesAlgo(u1 ,l1, d1, rhs1, p1, N);
    }

    time = clock() - time;

    printf("Serial time cost: %.2f secomds.\n", static_cast<double>(time)/CLOCKS_PER_SEC);

    fp = fopen(argv[2], "w+");

    for (int i=0;i<N;++i) {
        fprintf(
            fp, "%d %lf %lf %lf %lf\n",
            id1[i], u1[i], l1[i], rhs1[i], d1[i]);
    }

    fclose(fp);

    delete[] id; delete[] u; delete[] l;
    delete[] d; delete[] rhs; delete[] p;

    delete[] id1; delete[] u1; delete[] l1;
    delete[] d1; delete[] rhs1; delete[] p1;
}
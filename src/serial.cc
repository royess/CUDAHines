#include <cstdio>

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

    int *id;
    double *u;
    double *l;
    double *d;
    double *rhs;
    int *p;
    int N;
    
    fp = fopen(argv[1], "r");

    fscanf(fp, "%d", &N);

    id = new int [N];
    u = new double [N];
    l = new double [N];
    d = new double [N];
    rhs = new double [N];
    p = new int [N];

    for (int i=0;i<N;++i) {
        fscanf(
            fp, "%d %lf %lf %lf %lf %d",
            &id[i], &u[i], &l[i], &rhs[i], &d[i], &p[i]);
    }

    fclose(fp);

    HinesAlgo(u ,l, d, rhs, p, N);

    fp = fopen(argv[2], "w+");

    for (int i=0;i<N;++i) {
        fprintf(
            fp, "%d %lf %lf %lf %lf\n",
            id[i], u[i], l[i], rhs[i], d[i]);
    }

    fclose(fp);

    delete[] id;
    delete[] u;
    delete[] l;
    delete[] d;
    delete[] rhs;
    delete[] p;
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "mpi.h"
#include <omp.h>

int main(int argc, char *argv[])
{
    int nprocs, mype;
    MPI_Status stat;
    int order = 1;
    
    MPI_Init(&argc, &argv);

    // Number of communicators
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);

    // The rank of this mpi
    MPI_Comm_rank (MPI_COMM_WORLD, &mype);

    int N, NT, n1, nt, method;
    double L, T, u, v;

    // Declaring everything for the first rank
    if (mype == 0)
    {
        N = atoi(argv[1]);
        NT = atoi(argv[2]);
        L = atof(argv[3]);
        T = atof(argv[4]);
        u = atof(argv[5]);
        v = atof(argv[6]);
        nt = atoi(argv[7]);
        method = atoi(argv[8]);
        // Printing the inputs given by the user
        printf("The value of Matrix Dimension, N is:                  %i\n", N);
        printf("The value of Number of timestamps, NT is:             %i\n", NT);
        printf("The value of Physical Cartesian Domain Length, L is:  %f\n", L);
        printf("The value of Total Physical Timespan, T is:           %f\n", T);
        printf("The value of X velocity scalar, u is:                 %.10f\n", u);
        printf("The value of Y velocity scalar, v is:                 %.10f\n", v);
        int mem = N * N * 2;
        printf("Memory to be used will be %d time size of double\n", mem);

        // Broadcasting the values from rank 0 to other ranks
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&NT, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&L, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&u, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&v, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&method, 1, MPI_INT, 0, MPI_COMM_WORLD);

    }

    else
    {
        // Receiving broadcasted values from rank 0
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&NT, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&L, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&u, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&v, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&method, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (method == 3)
        order = 2;

    // Setting threads for OMP
    omp_set_num_threads(nt);

    // Calculating delta x and delta t
    double delx = L / N;
    double delt = T / NT;

    // Asserting courant stability condition
    assert(delt <= delx / sqrt(2 * (pow(u, 2) + pow(v, 2))));
    
    double x0 = L / 2;
    double y0 = x0;

    // Calculating sigma x and sigma y
    double sigx = L / 4;
    double sigy = sigx;

    n1 = sqrt(nprocs);
    int new_N = N / n1;

    // Start i and j for this rank
    int start_j = new_N * (mype % n1);
    int start_i = mype / n1;
    start_i *= new_N;  

    double *ptr;
    double *ptr1;
    double **mCn;
    double **mCn1, **temp;
    int len = sizeof(double *) * (new_N + order * 2) + sizeof(double) * (new_N + order * 2) * (new_N + order * 2);
    mCn = (double **)malloc(len);
    mCn1 = (double **)malloc(len);
    ptr = (double *)(mCn + new_N + order * 2);
    ptr1 = (double *)(mCn1 + new_N + order * 2);

    // Assigning row sizes to mCn and mCn1
    for (int i = 0; i < new_N + order * 2; i++)
    {
        mCn[i] = (ptr + (new_N + order * 2) * i);
        mCn1[i] = (ptr1 + (new_N + order * 2) * i);
    }

    // Assigning initial values to mCn
    for (int i = order; i < new_N + order; i++)
    {
        for (int j = order; j < new_N + order; j++)
        {
            mCn[i][j] = exp(
                -(
                    pow((i - order + start_i) * delx - x0, 2) / (2 * pow(sigx, 2)) + 
                    pow((j - order + start_j) * delx - y0 ,2) / (2 * pow(sigy, 2))
                )
            );
        }
    }

    // Creating arrays to pass the messages
    double for_send_top[(new_N + order * 2) * order];
    double for_send_bot[(new_N + order * 2) * order];
    double for_send_lef[(new_N + order * 2) * order];
    double for_send_rig[(new_N + order * 2) * order];

    double t1 = omp_get_wtime();
    
    for (int n = 0; n < NT; n++)
    {
        
        if ((mype % n1 + mype / n1) % 2 == 1)
        {   
            // Setting receive first for these cases
            MPI_Recv(for_send_lef, (new_N + order * 2) * order, MPI_DOUBLE, ((mype) % n1 ? mype - 1 : mype + n1 - 1) , 1, MPI_COMM_WORLD, &stat);
            MPI_Recv(for_send_rig, (new_N + order * 2) * order, MPI_DOUBLE, ((mype + 1) % n1 ? mype + 1 : mype - n1 + 1) , 1, MPI_COMM_WORLD, &stat);
            MPI_Recv(for_send_top, (new_N + order * 2) * order, MPI_DOUBLE, ((mype - n1) >= 0 ? mype - n1 : mype + nprocs - n1) , 1, MPI_COMM_WORLD, &stat);
            MPI_Recv(for_send_bot, (new_N + order * 2) * order, MPI_DOUBLE, ((mype + n1) < nprocs ? mype + n1 : mype - nprocs + n1) , 1, MPI_COMM_WORLD, &stat);     

            // Parallelising the for loop
            #ifdef PARALLEL
            #pragma omp parallel for default(none) shared(for_send_lef, for_send_rig, for_send_bot, for_send_top, n1, new_N, order, temp, mype, nprocs, N, NT, L, T, u, v, method, mCn, mCn1, delx, delt, x0, y0, sigx, sigy, n) schedule(static)
            #endif
            for (int i = order; i < (new_N + order * 2) * order - 1; i++)
            {
                mCn[i / (new_N + order * 2)][i % ((new_N + order * 2))] = for_send_top[i];
                mCn[i / (new_N + order * 2) + (new_N + order)][i % ((new_N + order * 2))] = for_send_bot[i];
                mCn[i % ((new_N + order * 2))][i / (new_N + order * 2)] = for_send_lef[i];
                mCn[i % ((new_N + order * 2))][i / (new_N + order * 2) + (new_N + order)] = for_send_rig[i];
                for_send_lef[i] = mCn[i % ((new_N + order * 2))][order + i / (new_N + order * 2)];
                for_send_rig[i] = mCn[i % ((new_N + order * 2))][i / (new_N + order * 2) + (new_N)];
                for_send_top[i] = mCn[order + i / (new_N + order * 2)][i % ((new_N + order * 2))];
                for_send_bot[i] = mCn[i / (new_N + order * 2) + (new_N)][i % ((new_N + order * 2))];
            }
            MPI_Send(for_send_bot, (new_N + order * 2) * order, MPI_DOUBLE, ((mype + n1) < nprocs ? mype + n1 : mype - nprocs + n1) , 1, MPI_COMM_WORLD);
            MPI_Send(for_send_top, (new_N + order * 2) * order, MPI_DOUBLE, ((mype - n1) >= 0 ? mype - n1 : mype + nprocs - n1), 1, MPI_COMM_WORLD);
            MPI_Send(for_send_rig, (new_N + order * 2) * order, MPI_DOUBLE, ((mype + 1)  % n1 ? mype + 1 : mype - n1 + 1), 1, MPI_COMM_WORLD);
            MPI_Send(for_send_lef, (new_N + order * 2) * order, MPI_DOUBLE, ((mype) % n1 ? mype - 1 : mype + n1 - 1), 1, MPI_COMM_WORLD);
            
        }
        else
        {
            // Setting send first for these cases
            #ifdef PARALLEL
            #pragma omp parallel for default(none) shared(for_send_lef, for_send_rig, for_send_bot, for_send_top, n1, new_N, order, temp, mype, nprocs, N, NT, L, T, u, v, method, mCn, mCn1, delx, delt, x0, y0, sigx, sigy, n) schedule(static)
            #endif
            for (int i = order; i < (new_N + order * 2) * order - 1; i++)
            {
                for_send_lef[i] = mCn[i % ((new_N + order * 2))][order + i / (new_N + order * 2)];
                for_send_rig[i] = mCn[i % ((new_N + order * 2))][i / (new_N + order * 2) + (new_N)];
                for_send_top[i] = mCn[order + i / (new_N + order * 2)][i % ((new_N + order * 2))];
                for_send_bot[i] = mCn[i / (new_N + order * 2) + (new_N)][i % ((new_N + order * 2))];
            }
            MPI_Send(for_send_bot, (new_N + order * 2) * order, MPI_DOUBLE, ((mype + n1) < nprocs ? mype + n1 : mype - nprocs + n1) , 1, MPI_COMM_WORLD);
            MPI_Send(for_send_top, (new_N + order * 2) * order, MPI_DOUBLE, ((mype - n1) >= 0 ? mype - n1 : mype + nprocs - n1), 1, MPI_COMM_WORLD);
            MPI_Send(for_send_rig, (new_N + order * 2) * order, MPI_DOUBLE, ((mype + 1)  % n1 ? mype + 1 : mype - n1 + 1), 1, MPI_COMM_WORLD);
            MPI_Send(for_send_lef, (new_N + order * 2) * order, MPI_DOUBLE, ((mype) % n1 ? mype - 1 : mype + n1 - 1), 1, MPI_COMM_WORLD);
            MPI_Recv(for_send_lef, (new_N + order * 2) * order, MPI_DOUBLE, ((mype) % n1 ? mype - 1 : mype + n1 - 1) , 1, MPI_COMM_WORLD, &stat);
            MPI_Recv(for_send_rig, (new_N + order * 2) * order, MPI_DOUBLE, ((mype + 1)  % n1 ? mype + 1 : mype - n1 + 1) , 1, MPI_COMM_WORLD, &stat);
            MPI_Recv(for_send_top, (new_N + order * 2) * order, MPI_DOUBLE, ((mype - n1) >= 0 ? mype - n1 : mype + nprocs - n1) , 1, MPI_COMM_WORLD, &stat);
            MPI_Recv(for_send_bot, (new_N + order * 2) * order, MPI_DOUBLE, ((mype + n1) < nprocs ? mype + n1 : mype - nprocs + n1) , 1, MPI_COMM_WORLD, &stat);

            #ifdef PARALLEL
            #pragma omp parallel for default(none) shared(for_send_lef, for_send_rig, for_send_bot, for_send_top, n1, new_N, order, temp, mype, nprocs, N, NT, L, T, u, v, method, mCn, mCn1, delx, delt, x0, y0, sigx, sigy, n) schedule(static)
            #endif
            for (int i = order; i < (new_N + order * 2) * order - 1; i++)
            {
                mCn[i / (new_N + order * 2)][i % ((new_N + order * 2))] = for_send_top[i];
                mCn[i / (new_N + order * 2) + (new_N + order)][i % ((new_N + order * 2))] = for_send_bot[i];
                mCn[i % ((new_N + order * 2))][i / (new_N + order * 2)] = for_send_lef[i];
                mCn[i % ((new_N + order * 2))][i / (new_N + order * 2) + (new_N + order)] = for_send_rig[i];
            }
        }

        u = 0;
        v = 0;

        // Parallelising the main loop
        #ifdef PARALLEL
        #pragma omp parallel for default(none) shared(n1, new_N, order, temp, mype, nprocs, N, NT, L, T, u, v, method, mCn, mCn1, delx, delt, x0, y0, sigx, sigy, n) schedule(static)
        #endif
        for (int i = order; i < new_N + order; i++)
        {
            for (int j = order; j < new_N + order; j++)
            {
                u = i * 10.0e-10;
                v = j * 10.0e-10;
                // The boundary conditions. Wrapping around whenever encountering a boundary
                int im = i - 1;
                int imm = i - 2;
                int ip = i + 1;
                int ipp = i + 2;
                int jm = j - 1;
                int jmm = j - 2;
                int jp = j + 1;
                int jpp = j + 2;

                // Using either of the three methods to find advection

                // Lax Method
                if (method == 1)
                    mCn1[i][j] = 1.0 / 4.0 * (mCn[im][j] + mCn[ip][j] + mCn[i][jm] + mCn[i][jp]) - delt / (2.0 * delx) * (u * (mCn[ip][j] - mCn[im][j]) + v * (mCn[i][jp] - mCn[i][jm]));

                // First order method
                else if (method == 2)
                {
                    if (u > 0 && v > 0)
                        mCn1[i][j] = mCn[i][j] - delt / delx * (u * (mCn[i][j] - mCn[im][j]) + v * (mCn[i][j] - mCn[i][jm]));
                    else
                        mCn1[i][j] = mCn[i][j] - delt / delx * (u * (mCn[ip][j] - mCn[i][j]) + v * (mCn[i][jp] - mCn[i][j]));
                }

                // Second order method
                else if (method == 3)
                {
                    if (u > 0 && v > 0)
                        mCn1[i][j] = mCn[i][j] - delt / (2.0 * delx) * (u * (3.0 * mCn[i][j] - 4.0 * mCn[im][j] + mCn[imm][j]) + (v * (3.0 * mCn[i][j] - 4.0 * mCn[i][jm] + mCn[i][jmm])));
                    else
                        mCn1[i][j] = mCn[i][j] + delt / (2.0 * delx) * (u * (3.0 * mCn[i][j] - 4.0 * mCn[ip][j] + mCn[ipp][j]) + (v * (3.0 * mCn[i][j] - 4.0 * mCn[i][jp] + mCn[i][jpp])));
                }
            }
        }

        // Swapping mCn and mCn1
        temp = mCn1;
        mCn1 = mCn;
        mCn = temp;
    }

    double t2 = omp_get_wtime();
    printf("time(s): %f\n", t2 - t1);

    // Syncing data to rank 0
    if (mype == 0)
    {
        double **Cn = (double**)malloc(N * sizeof(double*));

        for (int i = 0; i < N; i++)
            Cn[i] = (double*)malloc(N * sizeof(double));

        
        for (int k = 0; k < new_N; k++)
            for (int j = 0; j < new_N; j++)
                Cn[k][j] = mCn1[k + order][j + order];

        for (int i = 1; i < nprocs; i++)
        {
            MPI_Recv(&mCn1[0][0], pow((new_N + order * 2), 2), MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &stat);
            int start_j = new_N * (i % n1);
            int start_i = i / n1;
            start_i *= new_N; 

            for (int k = 0; k < new_N; k++)
                for (int j = 0; j < new_N; j++)
                    Cn[k + start_i][j + start_j] = mCn1[k + order][j + order];
        }

        for (int i = 0; i < N; i++)
            free(Cn[i]);

        free(Cn);
    }
    
    // If the rank is not 0
    else
        MPI_Send(&mCn1[0][0], pow((new_N + order * 2), 2), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

    free(mCn);
    free(mCn1);

    MPI_Finalize();

    return 0;
}
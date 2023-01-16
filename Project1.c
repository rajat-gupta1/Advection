#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

int main(int argc, char *argv[])
{
    int N, NT;
    double L, T, u, v;
    N = atoi(argv[1]);
    NT = atoi(argv[2]);
    L = atof(argv[3]);
    T = atof(argv[4]);
    u = atof(argv[5]);
    v = atof(argv[6]);

    // Printing the inputs given by the user
    printf("The value of Matrix Dimension, N is:                  %i\n", N);
    printf("The value of Number of timestamps, NT is:             %i\n", NT);
    printf("The value of Physical Cartesian Domain Length, L is:  %f\n", L);
    printf("The value of Total Physical Timespan, T is:           %f\n", T);
    printf("The value of X velocity scalar, u is:                 %.10f\n", u);
    printf("The value of Y velocity scalar, v is:                 %.10f\n", v);

    int mem = N * N * 2;
    printf("Memory to be used will be %d time size of double\n", mem);

    // Creating a 2D array using pointers
    double **Cn = (double**)malloc(N * sizeof(double*));
    double **Cn1 = (double**)malloc(N * sizeof(double*));
    double **temp;

    for (int i = 0; i < N; i++)
    {
        Cn1[i] = (double*)malloc(N * sizeof(double));
        Cn[i] = (double*)malloc(N * sizeof(double));
    }

    double delx = L / N;
    double delt = T / NT;

    assert(delt <= delx / sqrt(2 * (pow(u, 2) + pow(v, 2))));
    
    double x0 = L / 2;
    double y0 = x0;

    double sigx = L / 4;
    double sigy = sigx;

    // Initialising Cn using 2D Gaussian pulse initial condition
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            
            Cn[i][j] = exp(
                -(
                    pow(i * delx - x0, 2) / (2 * pow(sigx, 2)) + 
                    pow(j * delx - y0 ,2) / (2 * pow(sigy, 2))
                )
            );
        }

    for (int n = 0; n < NT; n++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                // The boundary conditions. Wrapping around whenever encountering a boundary
                int im = i - 1;
                int ip = i + 1;
                int jm = j - 1;
                int jp = j + 1;
                if (i == 0)
                    im = N - 1;
                if (j == 0)
                    jm = N - 1;
                if (i == N - 1)
                    ip = 0;
                if (j == N - 1)
                    jp = 0;

                // Updating Cn1
                Cn1[i][j] = 1.0 / 4.0 * (Cn[im][j] + Cn[ip][j] + Cn[i][jm] + Cn[i][jp]) - delt / (2.0 * delx) * (u * (Cn[ip][j] - Cn[im][j]) + v * (Cn[i][jp] - Cn[i][jm]));
            }
        }

        char output[20];

        // Saving file name into a variable output for the cases we need to output
        if (n == 0)
            strcpy(output,"output");
        else if (n == NT / 2)
            strcpy(output,"output2");
        else if (n == NT - 1)
            strcpy(output,"output3");

        // Saving output to output files
        if (n == 0 || n == NT / 2 || n == NT - 1)
        {
            FILE *ofp;
    
            if ((ofp = fopen(output, "w")) == NULL) {
                puts("Error: output file invalid");
                return -1;
            }

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                    fprintf(ofp, "%e,", Cn[i][j]);
                fprintf(ofp, "\n");
            }

            fclose(ofp);

        }

        // Storing value of Cn1 to Cn. Swapping it to ensure that we dont lose reference to the pointer of Cn1
        temp = Cn1;
        Cn1 = Cn;
        Cn = temp;
    }

    // Freeing up memory
    for (int i = 0; i < N; i++)
    {
        free(Cn[i]);
        free(Cn1[i]);
    }

    free(Cn);
    free(Cn1);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <math.h>
#include <string.h>
#include <Accelerate/Accelerate.h>

float *create_Imatrix(int N)
{
    float *I_matrix = (float *) malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i == j)
            {
                I_matrix[i * N + j] = 1;
            } else
            {
                I_matrix[i * N + j] = 0;
            }
        }
    }
    return I_matrix;
}

void print_matrix(float *matrix, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

//максимальная сумма строки
float get_max_line_sum(float *matrix, int N)
{
    float *sums = (float *) malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i)
    {
        sums[i] = cblas_sasum(N, matrix + N * i, 1);
        // вычисляет сумму абсолютных значений элементов в векторе

    }
    float result = sums[cblas_isamax(N, sums, 1)];
    //Возвращает индекс элемента с наибольшим абсолютным значением в векторе

    free(sums);
    return result;
}

//максимальная сумма столбца
float get_max_column_sum(float *matrix, int N)
{
    float *sums = (float *) malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i)
    {
        sums[i] = cblas_sasum(N, matrix + i, N);
        // вычисляет сумму абсолютных значений элементов в векторе

    }
    float result = sums[cblas_isamax(N, sums, 1)];
    //Возвращает индекс элемента с наибольшим абсолютным значением в векторе

    free(sums);
    return result;
}

int main()
{
    int N, M;

    printf("Размер матрицы: ");
    scanf("%d", &N);
    printf("\n");
    printf("Число членов ряда: ");
    scanf("%d", &M);
    printf("\n");

    srand(time(NULL));
    float *A_matrix = (float *) malloc(N * N * sizeof(float));

    //рандомное заполнение матрицы
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A_matrix[i * N + j] = (float) (rand() % 100);
        }
    }

    printf("Исходная матрица: \n");
    print_matrix(A_matrix, N);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    float *I_matrix = create_Imatrix(N);
    float *R_matrix = (float *) malloc(N * N * sizeof(float));
    float *previous = (float *) malloc(N * N * sizeof(float));
    float *current_sum = (float *) malloc(N * N * sizeof(float));
    float *sqr = (float *) malloc(N * N * sizeof(float));

    float max_line = get_max_line_sum(A_matrix, N);
    float max_column = get_max_column_sum(A_matrix, N);

    // R = -BA =-1 * A * AT / (|max_column| * |max_line|)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, -(1 / (max_line * max_column)), A_matrix, N, A_matrix,
                N, 0,
                R_matrix, N);
    // R = R + 1.0f * I
    cblas_saxpy(N * N, 1.0f, I_matrix, 1, R_matrix, 1);
    //prev = R
    memcpy(previous, R_matrix, N * N * sizeof(float));

    for (int i = 0; i < M; i++)
    {
        // C ← αAB + βC
        // sum = sum + prev
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                    1.0f, previous, N, I_matrix, N, 1, current_sum, N);

        //sqr = 0; sqr = prev * R
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                    1.0f, previous, N, R_matrix, N, 0, sqr, N);

        //обновляем prev
        memcpy(previous, sqr, N * N * sizeof(float));

    }

    // sum = I + (R + R^2 + ...)
    cblas_saxpy(N * N, 1.0f, I_matrix, 1, current_sum, 1);
    float *reverse_matrix = (float *) malloc(N * N * sizeof(float));

    // reverse = B * sum = AT * sum / (|max_column| * |max_line|)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N,
                (1 / (max_line * max_column)), current_sum, N, A_matrix, N, 0, reverse_matrix, N);



    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    double time = end.tv_sec - start.tv_sec + 0.000000001 * (end.tv_nsec - start.tv_nsec);

    printf("Обратная матрица: \n");
    print_matrix(reverse_matrix, N);

    printf("Time taken: %lf\n", time);

    free(A_matrix);
    free(I_matrix);
    free(R_matrix);
    free(previous);
    free(current_sum);
    free(reverse_matrix);
    free(sqr);

    return 0;
}


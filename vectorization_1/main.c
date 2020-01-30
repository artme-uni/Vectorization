#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define MAX(a, b) ((a > b) ? a : b)

//печатаем матрицу
void print_matrix(float *matrix, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%10f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
    return;
}

//умножение матрицы на число
void multiply_matrix_by_number(float *matrix, int N, float number)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i * N + j] = matrix[i * N + j] * number;
        }
    }
    return;
}

//сложение матриц
void add_matrices(float *X_matrix, float *Y_matrix, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            Y_matrix[i * N + j] = X_matrix[i * N + j] + Y_matrix[i * N + j];
        }
    }
    return;
}

//перемножение матриц
void multiply_matrices(float *A_matrix, float *B_matrix, float *result, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i * N + j] = 0;
            for (int k = 0; k < N; k++)
            {
                result[i * N + j] += (A_matrix[i * N + k] * B_matrix[k * N + j]);
            }
        }
    }
    return;
}

//создание единичной матрицы
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

//транспонируем матрицу
float *transpose_matrix(float *A_matrix, int N)
{
    float *AT_matrix = (float *) malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            AT_matrix[i * N + j] = A_matrix[j * N + i];
        }
    }
    return AT_matrix;
}

//считаем матрицу B = AT / (|max_column| * |max_line|)
float *calculate_Bmatrix(float *A_matrix, float *AT_matrix, int N)
{
    float max_line = 0;
    float max_column = 0;
    float line_sum = 0;
    float column_sum = 0;

    // ищем максимальную строку и столбец
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            line_sum += A_matrix[i * N + j];
            column_sum += A_matrix[j * N + i];
        }

        max_line = MAX(max_line, line_sum);
        max_column = MAX(max_column, column_sum);

        line_sum = 0;
        column_sum = 0;
    }

    multiply_matrix_by_number(AT_matrix, N, 1 / fabs(max_line * max_column));
    return AT_matrix;
}

//считаем матрицу R = I - AB
float *calculate_Rmatrix(float *A_matrix, float *B_matrix, float *I_matrix, int N)
{
    float *R_matrix = (float *) malloc(N * N * sizeof(float));
    multiply_matrices(B_matrix, A_matrix, R_matrix, N);
    multiply_matrix_by_number(R_matrix, N, -1.0);
    add_matrices(I_matrix, R_matrix, N);
    return R_matrix;
}

//считаем обратную матрицу
float *calculate_reverse_matrix(float *I_matrix, float *B_matrix, float *R_matrix, int N, int M)
{
    float *temp = (float *) malloc(N * N * sizeof(float));
    float *result = (float *) malloc(N * N * sizeof(float));

    memcpy(temp, R_matrix, N * N * sizeof(float));
    add_matrices(I_matrix, result, N);

    for (int i = 0; i < M; i++)
    {
        add_matrices(temp, result, N);
        multiply_matrices(temp, R_matrix, temp, N);
    }

    multiply_matrices(result, B_matrix, result, N);

    free(temp);
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
    float *AT_matrix = transpose_matrix(A_matrix, N);
    float *B_matrix = calculate_Bmatrix(A_matrix, AT_matrix, N);
    float *R_matrix = calculate_Rmatrix(A_matrix, B_matrix, I_matrix, N);

    float *reverse_matrix = calculate_reverse_matrix(I_matrix, B_matrix, R_matrix, N, M);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    double time = end.tv_sec - start.tv_sec + 0.000000001 * (end.tv_nsec - start.tv_nsec);

    printf("Обратная матрица: \n");
    print_matrix(reverse_matrix, N);

    printf("Time taken: %lf\n", time);

    free(A_matrix);
    free(I_matrix);
    free(AT_matrix);
    free(R_matrix);
    free(reverse_matrix);

    return 0;
}
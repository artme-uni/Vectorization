#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <xmmintrin.h>
#include <immintrin.h>

#define MAX(a, b) ((a > b) ? a : b)

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

float *create_Imatrix(int N)
{
    float *I_matrix = (float *) _mm_malloc(N * N * sizeof(float), 4 * sizeof(float));
    memset(I_matrix, 0, N * N * sizeof(float));
    //заполняем матрицу нулями

    for (int i = 0; i < N; i++)
    {
        I_matrix[i * (N + 1)] = 1;
    }

    return I_matrix;
}

float *transpose_matrix(float *A_matrix, int N)
{
    float *AT_matrix = (float *) _mm_malloc(N * N * sizeof(float), 4 * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            AT_matrix[i * N + j] = A_matrix[j * N + i];
        }
    }
    return AT_matrix;
}

//умножение матрицы на число
void multiply_matrix_by_number(float *matrix, int N, float number)
{
    __m128 line, result;
    __m128 mmNumber = _mm_set1_ps(number);
    //заполняем каждую компоненту вектора чилом, на которое нужно умножить

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j += 4)
        {
            line = _mm_load_ps(matrix + i * N + j);
            //загружаем значения по выравненному адресу
            result = _mm_mul_ps(line, mmNumber);
            //умножаем строку на число и получаем частичные суммы
            _mm_store_ps(matrix + i * N + j, result);
            //записываем значения по выравненному адресу
        }
    }
    return;
}

//сложение матриц
void add_matrices(float *X_matrix, float *Y_matrix, int N)
{
    __m128 lineX, lineY;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j += 4)
        {
            lineX = _mm_load_ps(X_matrix + i * N + j);
            //загружаем значения из X по выравненному адресу
            lineY = _mm_load_ps(Y_matrix + i * N + j);
            //загружаем значения из Y по выравненному адресу
            _mm_store_ps(Y_matrix + i * N + j, _mm_add_ps(lineX, lineY));
            //записываем сумму значений по выравненному адресу
        }
    }
    return;
}

//умножение матриц
void multiply_matrices(float *A_matrix, float *B_matrix, float *result, int N)
{
    __m128 line_a, line_b, temp_product, sum;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j += 4)
        {
            sum = _mm_setzero_ps();
            //обнуление значений

            for (int k = 0; k < N; k++)
            {
                line_a = _mm_set1_ps(A_matrix[i * N + k]);
                //заполняем каждую компоненту вектора значением из столбца матрицы A
                line_b = _mm_load_ps(B_matrix + k * N + j);
                //загружаем строку из матрицы B

                temp_product = _mm_mul_ps(line_a, line_b);
                //перемножаем
                sum = _mm_add_ps(sum, temp_product);
                //складываем
            }

            _mm_store_ps(result + i * N + j, sum);
            //заполняем матрицу result
        }
    }
    return;
}

//нахождение максимальной суммы строк
float maximal_sum(float *matrix, int N)
{
    const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    //сначала заполняем векторное число единицами (старший, знаковый бит равен нулю) и приводим тип

    __m128 sum, buffer;
    float temp_sum, max_sum = 0;

    for (int i = 0; i < N; i++)
    {
        sum = _mm_setzero_ps();
        //обнуление значений

        for (int j = 0; j < N; j += 4)
        {
            buffer = _mm_load_ps(matrix + i * N + j);
            //загружаем значения по выравненному адресу
            buffer = _mm_and_ps(buffer, mask);
            //and с маской для "модуля"
            sum = _mm_add_ps(sum, buffer);
            //обновляем sum
        }

        buffer = _mm_movehl_ps(buffer, sum);
        sum = _mm_add_ps(sum, buffer);
        buffer = _mm_shuffle_ps(sum, sum, 1);
        sum = _mm_add_ss(sum, buffer);
        _mm_store_ss(&temp_sum, sum);

        /* преобразованиями добиваемся того, что сумма лежит в младшем значении
         * buffer   v   w   !y   !e
         * sum      y   e    z    k
         *
         * sum      xx  xx  y+z  e+k
         * buffer   xx  xx  xx   y+z
         *
         * sum      xx  xx  xx   y+z+k+e
        */

        max_sum = MAX(max_sum, temp_sum);
    }

    return max_sum;
}

//считаем матрицу B = AT / (|max_column| * |max_line|)
float *calculate_Bmatrix(float *A_matrix, float *AT_matrix, int N)
{
    float A1 = maximal_sum(A_matrix, N);
    float Ainf = maximal_sum(AT_matrix, N);
    multiply_matrix_by_number(AT_matrix, N, 1 / (A1 * Ainf));

    return AT_matrix;
}

//считаем матрицу R = I - AB
float *calculate_Rmatrix(float *A_matrix, float *B_matrix, float *I_matrix, int N)
{
    float *R_matrix = (float *) _mm_malloc(N * N * sizeof(float), 4 * sizeof(float));
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

    /*
    float tmp1_matrix[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float tmp2_matrix[] = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    float *res_matrix = (float *) malloc(16 * sizeof(float));

    multiply_matrices(tmp1_matrix, tmp2_matrix, res_matrix, 4);

    print_matrix(res_matrix, 4);

    */

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

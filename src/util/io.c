/**
 * @brief implement for file read and write utils
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */ 

#include "alphasparse/util/io.h"
#include "alphasparse/util/error.h"
#include "alphasparse_cpu.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


FILE *alpha_open(const char * filename, const char * modes){
    FILE* file = fopen(filename, modes);
    if(file == NULL){
        printf("file is not exist!!!\n");
    }
    return file;
}

void alpha_close(FILE *stream){
    fclose(stream);
}

void result_write(const char *path, const size_t ele_num,size_t ele_size, const void *data)
{
    FILE *ans = fopen(path, "w");
    if (ans == NULL)
    {
        printf("ans file open error!!!\n");
        exit(-1);
    }
    fwrite(&ele_num, sizeof(size_t), 1, ans);
    fwrite(data, ele_size, ele_num, ans);
    fclose(ans);
}

void alpha_read_coo(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p, ALPHA_INT *nnz_p, ALPHA_INT **row_index, ALPHA_INT **col_index, float **values)
{
    FILE *fp = alpha_open(file, "r");
    char buffer[BUFFER_SIZE];
    char *token;
    const char SYM[] = "symmetric";
    int issym = 0;
    int firstLine = 1;
    while (fgets(buffer, BUFFER_SIZE, fp))
    {
        if (firstLine)
        {
            if (buffer[0] == '%')
            {
                token = strtok(buffer, " \n");
                while (token != NULL)
                {
                    if (strcmp(token, SYM) == 0)
                    {
                        issym = 1;
                        break;
                    }
                    token = strtok(NULL, " \n");
                }
            }
            firstLine = 0;
        }
        if (buffer[0] != '%')
            break;
    }
    ALPHA_INT64 m, n, nnz, real_nnz, double_nnz;
    sscanf(buffer, "%ld %ld %ld\n", &m, &n, &nnz);
    real_nnz = 0;
    double_nnz = nnz << 1;
    *m_p = (ALPHA_INT)m;
    *n_p = (ALPHA_INT)n;
    ALPHA_INT *fake_row_index = alpha_malloc(double_nnz * sizeof(ALPHA_INT));
    ALPHA_INT *fake_col_index = alpha_malloc(double_nnz * sizeof(ALPHA_INT));
    float *fake_values = alpha_malloc(double_nnz * sizeof(float));
    for (ALPHA_INT64 i = 0; i < nnz; i++, real_nnz++)
    {
        ALPHA_INT64 row, col;
        float val = 1.f;
        fgets(buffer, BUFFER_SIZE, fp);
        token = strtok(buffer, " ");
        row = atol(token);
        token = strtok(NULL, " ");
        col = atol(token);
        token = strtok(NULL, " ");
        if (token != NULL)
        {
            val = (float)atof(token);
        }
        fake_row_index[real_nnz] = (ALPHA_INT)row - 1;
        fake_col_index[real_nnz] = (ALPHA_INT)col - 1;
        fake_values[real_nnz] = val;
        if (row != col && issym)
        {
            real_nnz++;
            fake_row_index[real_nnz] = (ALPHA_INT)col - 1;
            fake_col_index[real_nnz] = (ALPHA_INT)row - 1;
            fake_values[real_nnz] = val;
        }
    }
    *row_index = alpha_malloc(real_nnz * sizeof(ALPHA_INT));
    *col_index = alpha_malloc(real_nnz * sizeof(ALPHA_INT));
    *values = alpha_malloc(real_nnz * sizeof(float));
    *nnz_p = real_nnz;
    memcpy(*row_index, fake_row_index, sizeof(ALPHA_INT) * real_nnz);
    memcpy(*col_index, fake_col_index, sizeof(ALPHA_INT) * real_nnz);
    memcpy(*values, fake_values, sizeof(float) * real_nnz);
    alpha_free(fake_row_index);
    alpha_free(fake_col_index);
    alpha_free(fake_values);
    alpha_close(fp);
}

void alpha_read_coo_d(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p, ALPHA_INT *nnz_p, ALPHA_INT **row_index, ALPHA_INT **col_index, double **values)
{
    FILE *fp = alpha_open(file, "r");
    char buffer[BUFFER_SIZE];
    char *token;
    const char SYM[] = "symmetric";
    int issym = 0;
    int firstLine = 1;
    while (fgets(buffer, BUFFER_SIZE, fp))
    {
        if (firstLine)
        {
            if (buffer[0] == '%')
            {
                token = strtok(buffer, " \n");
                while (token != NULL)
                {
                    if (strcmp(token, SYM) == 0)
                    {
                        issym = 1;
                        break;
                    }
                    token = strtok(NULL, " \n");
                }
            }
            firstLine = 0;
        }
        if (buffer[0] != '%')
            break;
    }
    ALPHA_INT64 m, n, nnz, real_nnz, double_nnz;
    sscanf(buffer, "%ld %ld %ld\n", &m, &n, &nnz);
    real_nnz = 0;
    double_nnz = nnz << 1;
    *m_p = (ALPHA_INT)m;
    *n_p = (ALPHA_INT)n;
    ALPHA_INT *fake_row_index = alpha_malloc(double_nnz * sizeof(ALPHA_INT));
    ALPHA_INT *fake_col_index = alpha_malloc(double_nnz * sizeof(ALPHA_INT));
    double *fake_values = alpha_malloc(double_nnz * sizeof(double));
    for (ALPHA_INT64 i = 0; i < nnz; i++, real_nnz++)
    {
        ALPHA_INT64 row, col;
        double val = 1.f;
        fgets(buffer, BUFFER_SIZE, fp);
        token = strtok(buffer, " ");
        row = atol(token);
        token = strtok(NULL, " ");
        col = atol(token);
        token = strtok(NULL, " ");
        if (token != NULL)
        {
            val = (double)atof(token);
        }
        fake_row_index[real_nnz] = (ALPHA_INT)row - 1;
        fake_col_index[real_nnz] = (ALPHA_INT)col - 1;
        fake_values[real_nnz] = val;
        if (row != col && issym)
        {
            real_nnz++;
            fake_row_index[real_nnz] = (ALPHA_INT)col - 1;
            fake_col_index[real_nnz] = (ALPHA_INT)row - 1;
            fake_values[real_nnz] = val;
        }
    }
    *row_index = alpha_malloc(real_nnz * sizeof(ALPHA_INT));
    *col_index = alpha_malloc(real_nnz * sizeof(ALPHA_INT));
    *values = alpha_malloc(real_nnz * sizeof(double));
    *nnz_p = real_nnz;
    memcpy(*row_index, fake_row_index, sizeof(ALPHA_INT) * real_nnz);
    memcpy(*col_index, fake_col_index, sizeof(ALPHA_INT) * real_nnz);
    memcpy(*values, fake_values, sizeof(double) * real_nnz);
    alpha_free(fake_row_index);
    alpha_free(fake_col_index);
    alpha_free(fake_values);
    alpha_close(fp);
}

void alpha_read_coo_c(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p, ALPHA_INT *nnz_p, ALPHA_INT **row_index, ALPHA_INT **col_index, ALPHA_Complex8 **values)
{
    FILE *fp = alpha_open(file, "r");
    char buffer[BUFFER_SIZE];
    char *token;
    const char SYM[] = "symmetric";
    int issym = 0;
    int firstLine = 1;
    while (fgets(buffer, BUFFER_SIZE, fp))
    {
        if (firstLine)
        {
            if (buffer[0] == '%')
            {
                token = strtok(buffer, " \n");
                while (token != NULL)
                {
                    if (strcmp(token, SYM) == 0)
                    {
                        issym = 1;
                        break;
                    }
                    token = strtok(NULL, " \n");
                }
            }
            firstLine = 0;
        }
        if (buffer[0] != '%')
            break;
    }
    ALPHA_INT64 m, n, nnz, real_nnz, double_nnz;
    sscanf(buffer, "%ld %ld %ld\n", &m, &n, &nnz);
    real_nnz = 0;
    double_nnz = nnz << 1;
    *m_p = (ALPHA_INT)m;
    *n_p = (ALPHA_INT)n;
    ALPHA_INT *fake_row_index = alpha_malloc(double_nnz * sizeof(ALPHA_INT));
    ALPHA_INT *fake_col_index = alpha_malloc(double_nnz * sizeof(ALPHA_INT));
    ALPHA_Complex8 *fake_values = alpha_malloc(double_nnz * sizeof(ALPHA_Complex8));
    for (ALPHA_INT64 i = 0; i < nnz; i++, real_nnz++)
    {
        ALPHA_INT64 row, col;
        ALPHA_Complex8 val = {1.f, 1.f};
        fgets(buffer, BUFFER_SIZE, fp);
        token = strtok(buffer, " ");
        row = atol(token);
        token = strtok(NULL, " ");
        col = atol(token);
        token = strtok(NULL, " ");
        if (token != NULL)
        {
            val.real = (float)atof(token);
            val.imag = (float)atof(token);
        }
        fake_row_index[real_nnz] = (ALPHA_INT)row - 1;
        fake_col_index[real_nnz] = (ALPHA_INT)col - 1;
        fake_values[real_nnz] = val;
        if (row != col && issym)
        {
            real_nnz++;
            fake_row_index[real_nnz] = (ALPHA_INT)col - 1;
            fake_col_index[real_nnz] = (ALPHA_INT)row - 1;
            fake_values[real_nnz] = val;
        }
    }
    *row_index = alpha_malloc(real_nnz * sizeof(ALPHA_INT));
    *col_index = alpha_malloc(real_nnz * sizeof(ALPHA_INT));
    *values = alpha_malloc(real_nnz * sizeof(ALPHA_Complex8));
    *nnz_p = real_nnz;
    memcpy(*row_index, fake_row_index, sizeof(ALPHA_INT) * real_nnz);
    memcpy(*col_index, fake_col_index, sizeof(ALPHA_INT) * real_nnz);
    memcpy(*values, fake_values, sizeof(ALPHA_Complex8) * real_nnz);
    alpha_free(fake_row_index);
    alpha_free(fake_col_index);
    alpha_free(fake_values);
    alpha_close(fp);
}

void alpha_read_coo_z(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p, ALPHA_INT *nnz_p, ALPHA_INT **row_index, ALPHA_INT **col_index, ALPHA_Complex16 **values)
{
    FILE *fp = alpha_open(file, "r");
    char buffer[BUFFER_SIZE];
    char *token;
    const char SYM[] = "symmetric";
    int issym = 0;
    int firstLine = 1;
    while (fgets(buffer, BUFFER_SIZE, fp))
    {
        if (firstLine)
        {
            if (buffer[0] == '%')
            {
                token = strtok(buffer, " \n");
                while (token != NULL)
                {
                    if (strcmp(token, SYM) == 0)
                    {
                        issym = 1;
                        break;
                    }
                    token = strtok(NULL, " \n");
                }
            }
            firstLine = 0;
        }
        if (buffer[0] != '%')
            break;
    }
    ALPHA_INT64 m, n, nnz, real_nnz, double_nnz;
    sscanf(buffer, "%ld %ld %ld\n", &m, &n, &nnz);
    real_nnz = 0;
    double_nnz = nnz << 1;
    *m_p = (ALPHA_INT)m;
    *n_p = (ALPHA_INT)n;
    ALPHA_INT *fake_row_index = alpha_malloc(double_nnz * sizeof(ALPHA_INT));
    ALPHA_INT *fake_col_index = alpha_malloc(double_nnz * sizeof(ALPHA_INT));
    ALPHA_Complex16 *fake_values = alpha_malloc(double_nnz * sizeof(ALPHA_Complex16));
    for (ALPHA_INT64 i = 0; i < nnz; i++, real_nnz++)
    {
        ALPHA_INT64 row, col;
        ALPHA_Complex16 val = {1.f, 1.f};
        fgets(buffer, BUFFER_SIZE, fp);
        token = strtok(buffer, " ");
        row = atol(token);
        token = strtok(NULL, " ");
        col = atol(token);
        token = strtok(NULL, " ");
        if (token != NULL)
        {
            val.real = (double)atof(token);
            val.imag = (double)atof(token);
        }
        fake_row_index[real_nnz] = (ALPHA_INT)row - 1;
        fake_col_index[real_nnz] = (ALPHA_INT)col - 1;
        fake_values[real_nnz] = val;
        if (row != col && issym)
        {
            real_nnz++;
            fake_row_index[real_nnz] = (ALPHA_INT)col - 1;
            fake_col_index[real_nnz] = (ALPHA_INT)row - 1;
            fake_values[real_nnz] = val;
        }
    }
    *row_index = alpha_malloc(real_nnz * sizeof(ALPHA_INT));
    *col_index = alpha_malloc(real_nnz * sizeof(ALPHA_INT));
    *values = alpha_malloc(real_nnz * sizeof(ALPHA_Complex16));
    *nnz_p = real_nnz;
    memcpy(*row_index, fake_row_index, sizeof(ALPHA_INT) * real_nnz);
    memcpy(*col_index, fake_col_index, sizeof(ALPHA_INT) * real_nnz);
    memcpy(*values, fake_values, sizeof(ALPHA_Complex16) * real_nnz);
    alpha_free(fake_row_index);
    alpha_free(fake_col_index);
    alpha_free(fake_values);
    alpha_close(fp);
}

#ifdef __MKL__

void mkl_read_coo(const char *file, MKL_INT *m_p, MKL_INT *n_p, MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index, float **values)
{
    FILE *fp = alpha_open(file, "r");
    char buffer[BUFFER_SIZE];
    char *token;
    const char SYM[] = "symmetric";
    int issym = 0;
    int firstLine = 1;
    while (fgets(buffer, BUFFER_SIZE, fp))
    {
        if (firstLine)
        {
            if (buffer[0] == '%')
            {
                token = strtok(buffer, " \n");
                while (token != NULL)
                {
                    if (strcmp(token, SYM) == 0)
                    {
                        issym = 1;
                        break;
                    }
                    token = strtok(NULL, " \n");
                }
            }
            firstLine = 0;
        }
        if (buffer[0] != '%')
            break;
    }
    MKL_INT64 m, n, nnz, real_nnz, double_nnz;
    sscanf(buffer, "%lld %lld %lld\n", &m, &n, &nnz);
    real_nnz = 0;
    double_nnz = nnz << 1;
    *m_p = (MKL_INT)m;
    *n_p = (MKL_INT)n;
    MKL_INT *fake_row_index = alpha_malloc(double_nnz * sizeof(MKL_INT));
    MKL_INT *fake_col_index = alpha_malloc(double_nnz * sizeof(MKL_INT));
    float *fake_values = alpha_malloc(double_nnz * sizeof(float));
    for (MKL_INT64 i = 0; i < nnz; i++, real_nnz++)
    {
        MKL_INT64 row, col;
        float val = 1.f;
        fgets(buffer, BUFFER_SIZE, fp);
        token = strtok(buffer, " ");
        row = atol(token);
        token = strtok(NULL, " ");
        col = atol(token);
        token = strtok(NULL, " ");
        if (token != NULL)
        {
            val = (float)atof(token);
        }
        fake_row_index[real_nnz] = (MKL_INT)row - 1;
        fake_col_index[real_nnz] = (MKL_INT)col - 1;
        fake_values[real_nnz] = val;
        if (row != col && issym)
        {
            real_nnz++;
            fake_row_index[real_nnz] = (MKL_INT)col - 1;
            fake_col_index[real_nnz] = (MKL_INT)row - 1;
            fake_values[real_nnz] = val;
        }
    }
    *row_index = alpha_malloc(real_nnz * sizeof(MKL_INT));
    *col_index = alpha_malloc(real_nnz * sizeof(MKL_INT));
    *values = alpha_malloc(real_nnz * sizeof(float));
    *nnz_p = real_nnz;
    memcpy(*row_index, fake_row_index, sizeof(MKL_INT) * real_nnz);
    memcpy(*col_index, fake_col_index, sizeof(MKL_INT) * real_nnz);
    memcpy(*values, fake_values, sizeof(float) * real_nnz);
    alpha_free(fake_row_index);
    alpha_free(fake_col_index);
    alpha_free(fake_values);
    alpha_close(fp);
}

void mkl_read_coo_d(const char *file, MKL_INT *m_p, MKL_INT *n_p, MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index, double **values)
{
    FILE *fp = alpha_open(file, "r");
    char buffer[BUFFER_SIZE];
    char *token;
    const char SYM[] = "symmetric";
    int issym = 0;
    int firstLine = 1;
    while (fgets(buffer, BUFFER_SIZE, fp))
    {
        if (firstLine)
        {
            if (buffer[0] == '%')
            {
                token = strtok(buffer, " \n");
                while (token != NULL)
                {
                    if (strcmp(token, SYM) == 0)
                    {
                        issym = 1;
                        break;
                    }
                    token = strtok(NULL, " \n");
                }
            }
            firstLine = 0;
        }
        if (buffer[0] != '%')
            break;
    }
    MKL_INT64 m, n, nnz, real_nnz, double_nnz;
    sscanf(buffer, "%lld %lld %lld\n", &m, &n, &nnz);
    real_nnz = 0;
    double_nnz = nnz << 1;
    *m_p = (MKL_INT)m;
    *n_p = (MKL_INT)n;
    MKL_INT *fake_row_index = alpha_malloc(double_nnz * sizeof(MKL_INT));
    MKL_INT *fake_col_index = alpha_malloc(double_nnz * sizeof(MKL_INT));
    double *fake_values = alpha_malloc(double_nnz * sizeof(double));
    for (MKL_INT64 i = 0; i < nnz; i++, real_nnz++)
    {
        MKL_INT64 row, col;
        double val = 1.f;
        fgets(buffer, BUFFER_SIZE, fp);
        token = strtok(buffer, " ");
        row = atol(token);
        token = strtok(NULL, " ");
        col = atol(token);
        token = strtok(NULL, " ");
        if (token != NULL)
        {
            val = (double)atof(token);
        }
        fake_row_index[real_nnz] = (MKL_INT)row - 1;
        fake_col_index[real_nnz] = (MKL_INT)col - 1;
        fake_values[real_nnz] = val;
        if (row != col && issym)
        {
            real_nnz++;
            fake_row_index[real_nnz] = (MKL_INT)col - 1;
            fake_col_index[real_nnz] = (MKL_INT)row - 1;
            fake_values[real_nnz] = val;
        }
    }
    *row_index = alpha_malloc(real_nnz * sizeof(MKL_INT));
    *col_index = alpha_malloc(real_nnz * sizeof(MKL_INT));
    *values = alpha_malloc(real_nnz * sizeof(double));
    *nnz_p = real_nnz;
    memcpy(*row_index, fake_row_index, sizeof(MKL_INT) * real_nnz);
    memcpy(*col_index, fake_col_index, sizeof(MKL_INT) * real_nnz);
    memcpy(*values, fake_values, sizeof(double) * real_nnz);
    alpha_free(fake_row_index);
    alpha_free(fake_col_index);
    alpha_free(fake_values);
    alpha_close(fp);
}

void mkl_read_coo_c(const char *file, MKL_INT *m_p, MKL_INT *n_p, MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index, MKL_Complex8 **values)
{
    FILE *fp = alpha_open(file, "r");
    char buffer[BUFFER_SIZE];
    char *token;
    const char SYM[] = "symmetric";
    int issym = 0;
    int firstLine = 1;
    while (fgets(buffer, BUFFER_SIZE, fp))
    {
        if (firstLine)
        {
            if (buffer[0] == '%')
            {
                token = strtok(buffer, " \n");
                while (token != NULL)
                {
                    if (strcmp(token, SYM) == 0)
                    {
                        issym = 1;
                        break;
                    }
                    token = strtok(NULL, " \n");
                }
            }
            firstLine = 0;
        }
        if (buffer[0] != '%')
            break;
    }
    MKL_INT64 m, n, nnz, real_nnz, double_nnz;
    sscanf(buffer, "%lld %lld %lld\n", &m, &n, &nnz);
    real_nnz = 0;
    double_nnz = nnz << 1;
    *m_p = (MKL_INT)m;
    *n_p = (MKL_INT)n;
    MKL_INT *fake_row_index = alpha_malloc(double_nnz * sizeof(MKL_INT));
    MKL_INT *fake_col_index = alpha_malloc(double_nnz * sizeof(MKL_INT));
    MKL_Complex8 *fake_values = alpha_malloc(double_nnz * sizeof(MKL_Complex8));
    for (MKL_INT64 i = 0; i < nnz; i++, real_nnz++)
    {
        MKL_INT64 row, col;
        MKL_Complex8 val = {1.f, 1.f};
        fgets(buffer, BUFFER_SIZE, fp);
        token = strtok(buffer, " ");
        row = atol(token);
        token = strtok(NULL, " ");
        col = atol(token);
        token = strtok(NULL, " ");
        if (token != NULL)
        {
            val.real = (float)atof(token);
            val.imag = (float)atof(token);
        }
        fake_row_index[real_nnz] = (MKL_INT)row - 1;
        fake_col_index[real_nnz] = (MKL_INT)col - 1;
        fake_values[real_nnz] = val;
        if (row != col && issym)
        {
            real_nnz++;
            fake_row_index[real_nnz] = (MKL_INT)col - 1;
            fake_col_index[real_nnz] = (MKL_INT)row - 1;
            fake_values[real_nnz] = val;
        }
    }
    *row_index = alpha_malloc(real_nnz * sizeof(MKL_INT));
    *col_index = alpha_malloc(real_nnz * sizeof(MKL_INT));
    *values = alpha_malloc(real_nnz * sizeof(MKL_Complex8));
    *nnz_p = real_nnz;
    memcpy(*row_index, fake_row_index, sizeof(MKL_INT) * real_nnz);
    memcpy(*col_index, fake_col_index, sizeof(MKL_INT) * real_nnz);
    memcpy(*values, fake_values, sizeof(MKL_Complex8) * real_nnz);
    alpha_free(fake_row_index);
    alpha_free(fake_col_index);
    alpha_free(fake_values);
    alpha_close(fp);
}

void mkl_read_coo_z(const char *file, MKL_INT *m_p, MKL_INT *n_p, MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index, MKL_Complex16 **values)
{
    FILE *fp = alpha_open(file, "r");
    char buffer[BUFFER_SIZE];
    char *token;
    const char SYM[] = "symmetric";
    int issym = 0;
    int firstLine = 1;
    while (fgets(buffer, BUFFER_SIZE, fp))
    {
        if (firstLine)
        {
            if (buffer[0] == '%')
            {
                token = strtok(buffer, " \n");
                while (token != NULL)
                {
                    if (strcmp(token, SYM) == 0)
                    {
                        issym = 1;
                        break;
                    }
                    token = strtok(NULL, " \n");
                }
            }
            firstLine = 0;
        }
        if (buffer[0] != '%')
            break;
    }
    MKL_INT64 m, n, nnz, real_nnz, double_nnz;
    sscanf(buffer, "%lld %lld %lld\n", &m, &n, &nnz);
    real_nnz = 0;
    double_nnz = nnz << 1;
    *m_p = (MKL_INT)m;
    *n_p = (MKL_INT)n;
    MKL_INT *fake_row_index = alpha_malloc(double_nnz * sizeof(MKL_INT));
    MKL_INT *fake_col_index = alpha_malloc(double_nnz * sizeof(MKL_INT));
    MKL_Complex16 *fake_values = alpha_malloc(double_nnz * sizeof(MKL_Complex16));
    for (MKL_INT64 i = 0; i < nnz; i++, real_nnz++)
    {
        MKL_INT64 row, col;
        MKL_Complex16 val = {1.f, 1.f};
        fgets(buffer, BUFFER_SIZE, fp);
        token = strtok(buffer, " ");
        row = atol(token);
        token = strtok(NULL, " ");
        col = atol(token);
        token = strtok(NULL, " ");
        if (token != NULL)
        {
            val.real = (double)atof(token);
            val.imag = (double)atof(token);
        }
        fake_row_index[real_nnz] = (MKL_INT)row - 1;
        fake_col_index[real_nnz] = (MKL_INT)col - 1;
        fake_values[real_nnz] = val;
        if (row != col && issym)
        {
            real_nnz++;
            fake_row_index[real_nnz] = (MKL_INT)col - 1;
            fake_col_index[real_nnz] = (MKL_INT)row - 1;
            fake_values[real_nnz] = val;
        }
    }
    *row_index = alpha_malloc(real_nnz * sizeof(MKL_INT));
    *col_index = alpha_malloc(real_nnz * sizeof(MKL_INT));
    *values = alpha_malloc(real_nnz * sizeof(MKL_Complex16));
    *nnz_p = real_nnz;
    memcpy(*row_index, fake_row_index, sizeof(MKL_INT) * real_nnz);
    memcpy(*col_index, fake_col_index, sizeof(MKL_INT) * real_nnz);
    memcpy(*values, fake_values, sizeof(MKL_Complex16) * real_nnz);
    alpha_free(fake_row_index);
    alpha_free(fake_col_index);
    alpha_free(fake_values);
    alpha_close(fp);
}

#endif

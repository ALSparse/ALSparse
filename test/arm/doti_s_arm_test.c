#include <alphasparse.h>
#include <stdio.h>
#include <math.h>

float alpha_doti_plain(const ALPHA_INT n, const float * x, const ALPHA_INT *indx, const float *y, int thread_num)
{
    alpha_set_thread_num(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    float res = alphasparse_s_doti_plain(n, x, indx, y);

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_s_doti_plain");

    return res;
}

float alpha_doti(const ALPHA_INT n, const float * x, const ALPHA_INT *indx, const float *y, int thread_num)
{
    alpha_set_thread_num(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    float res = alphasparse_s_doti(n, x, indx, y);

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_s_doti");

    return res;
}

int main(int argc,const char *argv[])
{
    // args
    args_help(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    // return
    ALPHA_INT plain_n = 50;
    ALPHA_INT alpha_n = 50;

    ALPHA_INT *alpha_incx = alpha_memalign(sizeof(ALPHA_INT) * alpha_n, DEFAULT_ALIGNMENT);
    ALPHA_INT *plain_incx = alpha_memalign(sizeof(ALPHA_INT) * alpha_n, DEFAULT_ALIGNMENT);

    float *alpha_x = alpha_memalign(sizeof(float) * alpha_n, DEFAULT_ALIGNMENT);
    float *plain_x = alpha_memalign(sizeof(float) * alpha_n, DEFAULT_ALIGNMENT);
    alpha_fill_random_s(alpha_x, 1, alpha_n);
    alpha_fill_random_s(plain_x, 1, alpha_n);

    for(ALPHA_INT i = 0; i < alpha_n; i++)
    {
        plain_incx[i] = i * 20;
        alpha_incx[i] = i * 20;
    }

    float *alpha_y = alpha_memalign(sizeof(float) * alpha_n * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random_s(alpha_y, 1, alpha_n * 20);
    
    float alpha_res = alpha_doti(alpha_n, alpha_x, alpha_incx, alpha_y, thread_num);
    int status = 0;
    if (check)
    {
        float plain_res = alpha_doti_plain(plain_n, plain_x, plain_incx, alpha_y, thread_num);
        status = fabs(alpha_res - plain_res) > 1e-2;
        if(!status)
        {
            fprintf(stderr, "doti_s correct\n");
        }
        else
        {
            fprintf(stderr, "doti_s error\n");
            fprintf(stderr, "plain : %f, openspblas : %f\n",plain_res, alpha_res);
        }        
    }

    alpha_free(plain_incx);
    alpha_free(alpha_incx);
    alpha_free(alpha_y);

    alpha_free(plain_x);
    alpha_free(alpha_x);

    return status;
}
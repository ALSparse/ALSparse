#include <alphasparse.h>
#include <stdio.h>
#include <math.h>

double alpha_doti_plain(const ALPHA_INT n, const double * x, const ALPHA_INT *indx, const double *y, int thread_num)
{
    alpha_set_thread_num(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    double res = alphasparse_d_doti_plain(n, x, indx, y);

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_d_doti_plain");

    return res;
}

double alpha_doti(const ALPHA_INT n, const double * x, const ALPHA_INT *indx, const double *y, int thread_num)
{
    alpha_set_thread_num(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    double res = alphasparse_d_doti(n, x, indx, y);

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_d_doti");

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

    double *alpha_x = alpha_memalign(sizeof(double) * alpha_n, DEFAULT_ALIGNMENT);
    double *plain_x = alpha_memalign(sizeof(double) * alpha_n, DEFAULT_ALIGNMENT);
    alpha_fill_random_d(alpha_x, 1, alpha_n);
    alpha_fill_random_d(plain_x, 1, alpha_n);

    for(ALPHA_INT i = 0; i < alpha_n; i++)
    {
        plain_incx[i] = i * 20;
        alpha_incx[i] = i * 20;
    }

    double *alpha_y = alpha_memalign(sizeof(double) * alpha_n * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random_d(alpha_y, 1, alpha_n * 20);
    
    double alpha_res = alpha_doti(alpha_n, alpha_x, alpha_incx, alpha_y, thread_num);
    int status = 0;
    if (check)
    {
        double plain_res = alpha_doti_plain(plain_n, plain_x, plain_incx, alpha_y, thread_num);
        status = fabs(alpha_res - plain_res) > 1e-10;
        if(!status)
        {
            fprintf(stderr, "doti_d correct\n");
        }
        else
        {
            fprintf(stderr, "doti_d error\n");
        }        
    }

    alpha_free(plain_incx);
    alpha_free(alpha_incx);
    alpha_free(alpha_y);

    alpha_free(plain_x);
    alpha_free(alpha_x);

    return status;
}
#include <alphasparse.h>
#include <stdio.h>

static void alpha_gthrz(const ALPHA_INT n, double * x, const ALPHA_INT *indx, double *y, int thread_num)
{
    alpha_set_thread_num(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_d_gthrz(n, y, x, indx), "alpha_d_gthrz");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_d_gthrz");
}

static void alpha_gthrz_plain(const ALPHA_INT n, double * x, const ALPHA_INT *indx, double *y, int thread_num)
{
    alpha_set_thread_num(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_d_gthrz_plain(n, y, x, indx), "alpha_d_gthrz_plain");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_d_gthrz_plain");
}

int main(int argc,const char *argv[])
{
    // args
    args_help(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    // return
    ALPHA_INT alpha_n = 50;

    ALPHA_INT *alpha_incx = alpha_memalign(sizeof(ALPHA_INT) * alpha_n, DEFAULT_ALIGNMENT);

    double *alpha_x = alpha_memalign(sizeof(double) * alpha_n, DEFAULT_ALIGNMENT);
    double *alpha_x_plain = alpha_memalign(sizeof(double) * alpha_n, DEFAULT_ALIGNMENT);

    for(ALPHA_INT i = 0; i < alpha_n; i++)
    {
        alpha_incx[i] = i * 20;
    }

    double *alpha_y = alpha_memalign(sizeof(double) * alpha_n * 20, DEFAULT_ALIGNMENT);
    double *alpha_y_plain = alpha_memalign(sizeof(double) * alpha_n * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random_d(alpha_y, 1, alpha_n * 20);
    alpha_fill_random_d(alpha_y_plain, 1, alpha_n * 20);
    
    alpha_gthrz(alpha_n, alpha_x, alpha_incx, alpha_y, thread_num);
    int status = 0;
    if (check)
    {
        alpha_gthrz_plain(alpha_n, alpha_x_plain, alpha_incx, alpha_y_plain, thread_num);
        status = check_d(alpha_x_plain, alpha_n, alpha_x, alpha_n);
    }

    alpha_free(alpha_incx);
    alpha_free(alpha_x);
    alpha_free(alpha_x_plain);

    return status;
}
#include <alphasparse.h>
#include <stdio.h>

static void alpha_gthrz(const ALPHA_INT n, float * x, const ALPHA_INT *indx, float *y, int thread_num)
{
    alpha_set_thread_num(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_gthrz(n, y, x, indx), "alpha_s_gthrz");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_s_gthrz");
}

static void alpha_gthrz_plain(const ALPHA_INT n, float * x, const ALPHA_INT *indx, float *y, int thread_num)
{
    alpha_set_thread_num(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_gthrz_plain(n, y, x, indx), "alpha_s_gthrz_plain");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_s_gthrz_plain");
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

    float *alpha_x = alpha_memalign(sizeof(float) * alpha_n, DEFAULT_ALIGNMENT);
    float *alpha_x_plain = alpha_memalign(sizeof(float) * alpha_n, DEFAULT_ALIGNMENT);

    for(ALPHA_INT i = 0; i < alpha_n; i++)
    {
        alpha_incx[i] = i * 20;
    }

    float *alpha_y = alpha_memalign(sizeof(float) * alpha_n * 20, DEFAULT_ALIGNMENT);
    float *alpha_y_plain = alpha_memalign(sizeof(float) * alpha_n * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random_s(alpha_y, 1, alpha_n * 20);
    alpha_fill_random_s(alpha_y_plain, 1, alpha_n * 20);
    
    alpha_gthrz(alpha_n, alpha_x, alpha_incx, alpha_y, thread_num);
    int status = 0;
    if (check)
    {
        alpha_gthrz_plain(alpha_n, alpha_x_plain, alpha_incx, alpha_y_plain, thread_num);
        status = check_s(alpha_x_plain, alpha_n, alpha_x, alpha_n);
    }

    alpha_free(alpha_incx);
    alpha_free(alpha_x);
    alpha_free(alpha_x_plain);

    return status;
}
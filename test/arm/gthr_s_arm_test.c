#include <alphasparse.h>
#include <stdio.h>

static void plain_gthr(const ALPHA_INT n, float * x, const ALPHA_INT* indx, const float *y, int thread_num)
{
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alphasparse_s_gthr_plain(n, y, x, indx);

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_s_gthr_plain");
}

static void alpha_gthr(const ALPHA_INT n, float * x, const ALPHA_INT *indx, const float *y, int thread_num)
{
    alpha_set_thread_num(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_gthr(n, y, x, indx), "alpha_s_gthr");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_s_gthr");
}

int main(int argc,const char *argv[])
{
    // args
    args_help(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    const float alpha = 3.;

    // return
    ALPHA_INT plain_n = 50;
    float a = 2.0f;
    ALPHA_INT alpha_n = 50;

    ALPHA_INT *plain_incx = alpha_memalign(sizeof(ALPHA_INT) * plain_n, DEFAULT_ALIGNMENT);
    ALPHA_INT *alpha_incx = alpha_memalign(sizeof(ALPHA_INT) * alpha_n, DEFAULT_ALIGNMENT);

    float *plain_x = alpha_memalign(sizeof(float) * plain_n, DEFAULT_ALIGNMENT);
    float *alpha_x = alpha_memalign(sizeof(float) * alpha_n, DEFAULT_ALIGNMENT);
    // alpha_fill_random_s(x, 1, alpha_n);

    for(ALPHA_INT i = 0; i < alpha_n; i++)
    {
        plain_incx[i] = i * 20;
        alpha_incx[i] = i * 20;
    }

    float *alpha_y = alpha_memalign(sizeof(float) * alpha_n * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random_s(alpha_y, 1, alpha_n * 20);
    
    alpha_gthr(alpha_n, alpha_x, alpha_incx, alpha_y, thread_num);
    int status = 0;
    if (check)
    {
        alpha_gthr(plain_n, plain_x, plain_incx, alpha_y, thread_num);
        status = check_s(plain_x, plain_n, alpha_x, alpha_n);
    }

    alpha_free(plain_incx);
    alpha_free(alpha_incx);
    alpha_free(alpha_y);

    alpha_free(plain_x);
    alpha_free(alpha_x);

    return status;
}
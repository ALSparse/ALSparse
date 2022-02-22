#include <alphasparse.h>
#include <stdio.h>

static void plain_axpyi(const ALPHA_INT n, const float a, const float * x, const ALPHA_INT* indx, float *y, int thread_num)
{
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alphasparse_s_axpy_plain(n, a, x, indx, y);

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_s_axpy_plain");
}

static void alpha_axpyi(const ALPHA_INT n, const float a, const float * x, const ALPHA_INT *indx, float *y, int thread_num)
{
    alpha_set_thread_num(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_axpy(n, a, x, indx, y), "alpha_s_axpyi");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_s_axpyi");
}

int main(int argc,const char *argv[])
{
    // args
    args_help(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    // return
    ALPHA_INT plain_n = 50;
    float a = 2.0f;
    ALPHA_INT alpha_n = 50;

    ALPHA_INT *plain_incx = alpha_memalign(sizeof(ALPHA_INT) * alpha_n, DEFAULT_ALIGNMENT);
    ALPHA_INT *alpha_incx = alpha_memalign(sizeof(ALPHA_INT) * alpha_n, DEFAULT_ALIGNMENT);

    float *x = alpha_memalign(sizeof(float) * alpha_n, DEFAULT_ALIGNMENT);
    alpha_fill_random_s(x, 1, alpha_n);

    for(ALPHA_INT i = 0; i < alpha_n; i++)
    {
        plain_incx[i] = i * 20;
        alpha_incx[i] = i * 20;
    }

    float *plain_y = alpha_memalign(sizeof(float) * alpha_n * 20, DEFAULT_ALIGNMENT);
    float *alpha_y = alpha_memalign(sizeof(float) * alpha_n * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random_s(plain_y, 1, alpha_n * 20);
    alpha_fill_random_s(alpha_y, 1, alpha_n * 20);
    
    alpha_axpyi(alpha_n, a, x, alpha_incx, alpha_y, thread_num);
    int status = 0;
    if (check)
    {
        plain_axpyi(plain_n, a, x, plain_incx, plain_y, thread_num);
        status = check_s(alpha_y, alpha_n * 20, plain_y, plain_n * 20);
    }

    alpha_free(plain_incx);
    alpha_free(alpha_incx);
    alpha_free(plain_y);
    alpha_free(alpha_y);

    return status;
}
#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

static void mkl_gthrz(const MKL_INT n, MKL_Complex8 * x, const MKL_INT* indx, MKL_Complex8 *y, int thread_num)
{
    mkl_set_num_threads(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    cblas_cgthrz(n, y, x, indx);

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "cblas_sgthrz");
}

static void alpha_gthrz(const ALPHA_INT n, ALPHA_Complex8 * x, const ALPHA_INT *indx, ALPHA_Complex8 *y, int thread_num)
{
    alpha_set_thread_num(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_c_gthrz_plain(n, y, x, indx), "alpha_c_gthrz");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_c_gthrz");
}

int main(int argc,const char *argv[])
{
    // args
    args_help(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    // return
    MKL_INT mkl_n = 50;
    ALPHA_INT alpha_n = 50;

    ALPHA_INT *alpha_incx = alpha_memalign(sizeof(ALPHA_INT) * alpha_n, DEFAULT_ALIGNMENT);
    MKL_INT *mkl_incx = alpha_memalign(sizeof(MKL_INT) * mkl_n, DEFAULT_ALIGNMENT);

    ALPHA_Complex8 *alpha_x = alpha_memalign(sizeof(ALPHA_Complex8) * alpha_n, DEFAULT_ALIGNMENT);
    MKL_Complex8 *mkl_x = alpha_memalign(sizeof(MKL_Complex8) * mkl_n, DEFAULT_ALIGNMENT);
    // alpha_fill_random_s(x, 1, alpha_n);

    for(ALPHA_INT i = 0; i < alpha_n; i++)
    {
        mkl_incx[i] = i * 20;
        alpha_incx[i] = i * 20;
    }

    ALPHA_Complex8 *alpha_y = alpha_memalign(sizeof(ALPHA_Complex8) * alpha_n * 20, DEFAULT_ALIGNMENT);
    MKL_Complex8 *mkl_y = alpha_memalign(sizeof(MKL_Complex8) * mkl_n * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random_c(alpha_y, 1, alpha_n * 20);
    alpha_fill_random_s((float *)mkl_y, 1, 2*mkl_n * 20);
    
    alpha_gthrz(alpha_n, alpha_x, alpha_incx, alpha_y, thread_num);
    int status = 0;
    if (check)
    {
        mkl_gthrz(mkl_n, mkl_x, mkl_incx, mkl_y, thread_num);
        status = check_s((float *)mkl_x, 2*mkl_n, (float *)alpha_x, 2*alpha_n);
    }

    return status;
}
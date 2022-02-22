#include <alphasparse.h>
#include <stdio.h>
#include <memory.h>

static void alpha_dotci_plain(const ALPHA_INT n, const ALPHA_Complex8 * x, const ALPHA_INT *indx, const ALPHA_Complex8 *y, ALPHA_Complex8 *dotci, int thread_num)
{
    alpha_set_thread_num(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alphasparse_c_dotci_sub_plain(n, x, indx, y, dotci);

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_c_dotci_plain");

    return;
}

static void alpha_dotci(const ALPHA_INT n, const ALPHA_Complex8 * x, const ALPHA_INT *indx, const ALPHA_Complex8 *y, ALPHA_Complex8 *dotci, int thread_num)
{
    alpha_set_thread_num(thread_num);
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alphasparse_c_dotci_sub(n, x, indx, y, dotci);

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_c_dotci");

    return;
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
    ALPHA_INT *plain_incx = alpha_memalign(sizeof(ALPHA_INT) * plain_n, DEFAULT_ALIGNMENT);

    ALPHA_Complex8 *alpha_x = alpha_memalign(sizeof(ALPHA_Complex8) * alpha_n, DEFAULT_ALIGNMENT);
    ALPHA_Complex8 *plain_x = alpha_memalign(sizeof(ALPHA_Complex8) * plain_n, DEFAULT_ALIGNMENT);
    alpha_fill_random_c(alpha_x, 1, alpha_n);
    alpha_fill_random_c(plain_x, 1, plain_n);

    for(ALPHA_INT i = 0; i < alpha_n; i++)
    {
        plain_incx[i] = i * 20;
        alpha_incx[i] = i * 20;
    }

    ALPHA_Complex8 *alpha_y = alpha_memalign(sizeof(ALPHA_Complex8) * alpha_n * 20, DEFAULT_ALIGNMENT);
    ALPHA_Complex8 *plain_y = alpha_memalign(sizeof(ALPHA_Complex8) * plain_n * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random_c(alpha_y, 1, alpha_n * 20);
    alpha_fill_random_c(plain_y, 1, alpha_n * 20);

    ALPHA_Complex8 * plain_dotci_v = alpha_memalign(sizeof(ALPHA_Complex8), DEFAULT_ALIGNMENT);
    ALPHA_Complex8 * alpha_dotci_v = alpha_memalign(sizeof(ALPHA_Complex8), DEFAULT_ALIGNMENT);

    memset(plain_dotci_v, 0, sizeof(ALPHA_Complex8));
    memset(alpha_dotci_v, 0, sizeof(ALPHA_Complex8));
    
    alpha_dotci(alpha_n, alpha_x, alpha_incx, alpha_y, alpha_dotci_v, thread_num);
    int status = 0;
    if (check)
    {
        alpha_dotci_plain(plain_n, plain_x, plain_incx, plain_y, plain_dotci_v, thread_num);
        status = check_c(alpha_dotci_v, 1, plain_dotci_v, 1);
    }

    alpha_free(plain_incx);
    alpha_free(alpha_incx);
    alpha_free(plain_y);
    alpha_free(alpha_y);
    alpha_free(plain_dotci_v);
    alpha_free(alpha_dotci_v);

    alpha_free(plain_x);
    alpha_free(alpha_x);

    return status;
}
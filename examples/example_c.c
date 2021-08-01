#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <stdml/tensor.h>

int64_t get_time_us()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return 1000000 * (uint64_t)tv.tv_sec + tv.tv_usec;
}

int main()
{
    tensor_t *x = stdml_new_tensor(i32, cpu, 2, 2, 3);
    // tensor_t *x = stdml_new_tensor(i32, gpu, 2, 2, 3);
    stdml_del_tensor(x);
    return 0;
}

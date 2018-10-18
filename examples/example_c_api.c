#include <stdlib.h>
#include <tensor.h>

int main()
{
    tensor_t *pt = new_tensor(0, 1, 1);
    del_tensor(pt);
    return 0;
}

#include <stdlib.h>
#include <tensor.h>

int main()
{
    {
        tensor_t *pt = new_tensor(dtypes.u8, 1, 1);
        del_tensor(pt);
    }
    {
        int dims[1] = {1};
        tensor_t *pt = new_tensor1(dtypes.u8, 1, dims);
        del_tensor(pt);
    }
    return 0;
}

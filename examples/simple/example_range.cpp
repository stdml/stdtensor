#include <iostream>
#include <ttl/range>

int main()
{
    for (auto i : ttl::range(10)) { std::cout << i << std::endl; }
    return 0;
}

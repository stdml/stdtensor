#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "testing.hpp"

namespace fs = std::filesystem;

int loc(const char *filename)
{
    FILE *fp = std::fopen(filename, "r");
    if (fp == nullptr) { return 0; }
    constexpr int max_line = 1 << 16;
    char line[max_line];
    int ln = 0;
    while (std::fgets(line, max_line - 1, fp)) { ++ln; }
    std::fclose(fp);
    return ln;
}

TEST(test_loc, test1)
{
    std::string path = "/path/to/directory";
    int tot = 0;
    int n = 0;
    for (const auto &entry : fs::directory_iterator("include/ttl/bits")) {
        const int ln = loc(entry.path().c_str());
        printf("%4d %s\n", ln, entry.path().c_str());
        ASSERT_TRUE(ln <= 200);
        tot += ln;
        ++n;
    }
    printf("total: %d lines in %d files\n", tot, n);
    ASSERT_TRUE(tot <= 2000);
}

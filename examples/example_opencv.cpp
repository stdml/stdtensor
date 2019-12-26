#include <opencv2/opencv.hpp>
#include <ttl/tensor>

using pixel_t = std::array<uint8_t, 3>;
static_assert(sizeof(pixel_t) == 3, "invalid pixel size");

using bmp_t = ttl::matrix<pixel_t>;

void save_bmp(const bmp_t &bmp)
{
    const cv::Mat img(cv::Size(bmp.shape().dims()[1], bmp.shape().dims()[0]),
                      CV_8UC(3), (void *)bmp.data());
    cv::imwrite("i.png", img);
}

void example_1(const int h, const int w)
{
    bmp_t img(h, w);  // Note that img is fully packed, without row padding
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            pixel_t &p = img[i][j];
            p[0] = i & 0xff;
            p[1] = j & 0xff;
            p[2] = (i + j) & 0xff;
        }
    }
    save_bmp(img);
}

int main()
{
    example_1(768, 1024);
    return 0;
}

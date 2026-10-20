// C++ check for repro 02: cv::bitwise_not on a CV_Bool Mat.
// Used to confirm the behaviour on the 5.x branch (opencv_core only), where the Python wheel is not available.
// Build against an OpenCV 5 tree, e.g.:  g++ -std=c++17 repro_02_bitwise_not_bool.cpp -I<opencv>/modules/core/include \
//     -I<build> -I<build>/modules/core -L<build>/lib -lopencv_core -Wl,-rpath,<build>/lib -o bool_not
#include <opencv2/core.hpp>
#include <cstdint>
#include <cstdio>

int main()
{
    printf("OpenCV %s\n", cv::getVersionString().c_str());
    uint8_t in_bytes[6] = {1, 0, 1, 0, 0, 1};
    cv::Mat src(2, 3, CV_Bool, in_bytes);
    cv::Mat dst;
    cv::bitwise_not(src, dst);
    printf("bitwise_not: type=%d (CV_Bool=%d) bytes:", dst.type(), (int)CV_Bool);
    for (int i = 0; i < 6; i++) printf(" %u", (unsigned)dst.data[i]);
    printf("   (a logical not would give 0 1 0 1 1 0)\n");

    uint8_t b_bytes[6] = {1, 1, 0, 0, 1, 0};
    cv::Mat b(2, 3, CV_Bool, b_bytes), r;
    cv::bitwise_and(src, b, r);
    printf("bitwise_and: bytes:");
    for (int i = 0; i < 6; i++) printf(" %u", (unsigned)r.data[i]);
    printf("   (expected 1 0 0 0 0 0)\n");
    return 0;
}

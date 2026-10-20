// C++ check that the 5.x branch no longer shows drafts 01 (MSER defaults) and 03 (remap with a mismatched map2).
// Built against opencv_core, opencv_imgproc, opencv_features of the 5.x branch (checked at commit 9123a3c209, 5.1.0-dev):
//   g++ -std=c++17 check_5x_fixes.cpp -I<src>/modules/{core,imgproc,features,flann}/include -I<build> -I<build>/modules/core \
//       -L<build>/lib -lopencv_features -lopencv_imgproc -lopencv_flann -lopencv_core -Wl,-rpath,<build>/lib
// Expected on 5.x: MSER finds 2 and 22 regions; both remap calls throw. On 5.0.0 MSER finds 0 and remap returns an image.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features.hpp>
#include <cstdio>
#include <vector>

int main()
{
    printf("OpenCV %s\n", cv::getVersionString().c_str());

    // --- MSER, image of issue 29652: filled rectangle with a circular hole
    cv::Mat img = cv::Mat::zeros(200, 200, CV_8U);
    cv::rectangle(img, cv::Point(40, 40), cv::Point(160, 160), cv::Scalar(255), -1);
    cv::circle(img, cv::Point(100, 100), 30, cv::Scalar(0), -1);
    {
        cv::Ptr<cv::MSER> m = cv::MSER::create();
        std::vector<std::vector<cv::Point> > regions; std::vector<cv::Rect> boxes;
        m->detectRegions(img, regions, boxes);
        printf("MSER default on rect+hole: %zu regions (4.14.0: 2, 5.0.0: 0); minDiversity=%.2f\n", regions.size(), m->getMinDiversity());
    }
    // --- MSER on the 25-rectangle texture of probes/api_diff.py is generated with numpy there; use rectangles of varied gray here
    {
        cv::Mat t(160, 200, CV_8U, cv::Scalar(30));
        cv::RNG rng(1);
        for (int i = 0; i < 25; i++) {
            int x = rng.uniform(10, 170), y = rng.uniform(10, 130), w = rng.uniform(8, 25), h = rng.uniform(8, 25);
            cv::rectangle(t, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(rng.uniform(80, 255)), -1);
        }
        cv::Ptr<cv::MSER> m = cv::MSER::create();
        std::vector<std::vector<cv::Point> > regions; std::vector<cv::Rect> boxes;
        m->detectRegions(t, regions, boxes);
        cv::Ptr<cv::MSER> m0 = cv::MSER::create(5, 60, 14400, 0.25, 0.0);
        std::vector<std::vector<cv::Point> > regions0; std::vector<cv::Rect> boxes0;
        m0->detectRegions(t, regions0, boxes0);
        printf("MSER default on 25 random rectangles: %zu regions; with min_diversity=0: %zu\n", regions.size(), regions0.size());
    }

    // --- remap with float32 map1 and a map2 of another type
    {
        cv::Mat src(6, 8, CV_8U, cv::Scalar(7)), dst;
        cv::Mat mx(6, 8, CV_32F, cv::Scalar(1)), my64(6, 8, CV_64F, cv::Scalar(1)), my8(6, 8, CV_8U, cv::Scalar(1));
        const char* names[] = {"float64", "uint8"};
        cv::Mat maps[] = {my64, my8};
        for (int i = 0; i < 2; i++) {
            try {
                cv::remap(src, dst, mx, maps[i], cv::INTER_LINEAR);
                printf("remap(float32 map1, %s map2): returned an image (no exception)\n", names[i]);
            } catch (const cv::Exception& e) {
                printf("remap(float32 map1, %s map2): threw cv::Exception code %d: %.90s\n", names[i], e.code, e.msg.c_str());
            }
        }
    }
    return 0;
}

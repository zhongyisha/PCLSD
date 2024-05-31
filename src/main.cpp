#include <iostream>
#include <opencv2/opencv.hpp>
#include "PCLSD.h"
#include <fstream>
#include <string>
#include <vector>
#include <time.h>

inline void
drawSegments(cv::Mat img,
             upm::Segments segs,
             const cv::Scalar &color,
             int thickness = 1,
             int lineType = cv::LINE_AA,
             int shift = 0) {
   // cv::Mat im = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
  for (const upm::Segment &seg: segs)
    cv::line(img, cv::Point2f(seg[0], seg[1]), cv::Point2f(seg[2], seg[3]), color, thickness, lineType, shift);
  cv::imshow("PCLSD", img);
  cv::waitKey();
  cv::destroyAllWindows();
}

void batched_process()
{
    clock_t start, finish;
    double sum_time = 0;
    upm::PCLSD pclsd;
    std::vector<std::string> results;
    cv::glob("E:\\Matlab2019b\\workSpace\\YorkUrban\\*.jpg", results, true);
  //  cv::glob("E:\\3D\\merton\\*.jpg", results, true);
   // cv::glob("E:\\Matlab2019b\\workSpace\\noiseImageH\\*.jpg", results, true);
  //  cv::glob("E:\\dataset\\wireframe\\test\\*.jpg", results, true);
  //  cv::glob("E:\\qq资料\\Evaluations\\Evaluations\\InputData\\Renoir-LineSegment\\TestImages\\*.bmp", results, true);
    std::sort(results.begin(), results.end());
    //std::cout << results[0] << std::endl;
    std::string log_dir = "E:\\vs2022\\workspace\\test\\logs\\";
    for (int i = 0; i < results.size(); ++i) {
        // read image
        cv::Mat img = cv::imread(results[i]);
        // process
        start = clock();
        upm::Segments segs = pclsd.detect(img);
        finish = clock();
        double elapsed_time = static_cast<double>(finish - start) / CLOCKS_PER_SEC;
        sum_time+= elapsed_time;

        std::cout << "PCLSD detected: " << segs.size() << " (large) segments" << std::endl;

        std::ofstream result(log_dir + results[i].substr(results[i].size() - 12, 8) + ".txt");
        for (int i = 0; i < segs.size(); ++i) {
            for (int j = 0; j < 4; ++j) {
                result << segs[i][j] << " ";
            }
            result << std::endl;
        }
    }
    sum_time /= results.size();
    std::cout <<"平均耗时:" << sum_time << std::endl;
}

int main() {
  
  //  batched_process();
  //  return 0;

   cv::Mat img = cv::imread("E:/vs2022/PCLSD/Pic/P1020824.jpg");
  if (img.empty()) {
    std::cerr << "Error reading input image" << std::endl;
    return -1;
  }
  upm::PCLSD pclsd;
  upm::Segments segs = pclsd.detect(img);
  std::cout << "PCLSD detected: " << segs.size() << " (large) segments" << std::endl;

  drawSegments(img, segs, CV_RGB(0, 255, 0), 1);

  return 0;
}
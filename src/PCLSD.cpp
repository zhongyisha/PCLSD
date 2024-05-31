#include "PCLSD.h"
#include "EdgeDrawer.h"

// Decides if we should take the image gradients as the interpolated version of the pixels right in the segment
// or if they are ready directly from the image
#define UPM_SD_USE_REPROJECTION


namespace upm {
    void PCLSD::drawPoint(cv::Mat img) {
        for (int i = 0; i < anchors.size(); i++) {
            cv::Point p;
            p.x = anchors[i].x;
            p.y = anchors[i].y;
            cv::circle(img, p, 1, cv::Scalar(0, 0, 255), -1);
        }
        cv::imshow("Point", img);
        cv::waitKey();
    }

    bool PCLSD::next(int& xSeed, int& ySeed) {
        int cols = maskImage.cols;
        int rows = maskImage.rows;

        uchar* ptrM = maskImage.data;
        uchar* ptrO = (uchar*)imgInfo->oriImg.data;

        int direction = ptrO[ySeed * cols + xSeed]; //(x,y)坐标的方向
        int direction0 = direction - 1;
        if (direction0 < 0) direction0 = 15;
        int direction1 = direction;
        int direction2 = direction + 1;
        if (direction2 == 16) direction2 = 0;

        static const int X_OFFSET[8] = { 0, 1, 0, -1, 1, -1, -1, 1 };
        static const int Y_OFFSET[8] = { 1, 0, -1, 0, 1, 1, -1, -1 };

        int x, y;
        for (size_t i = 0; i != 8; ++i) //（x,y）的8邻域
        {
            x = xSeed + X_OFFSET[i];
            if ((0 <= x) && (x < cols))
            {
                y = ySeed + Y_OFFSET[i];
                if ((0 <= y) && (y < rows))
                {
                    if (ptrM[y * cols + x])//标记图像中有意义的像素点
                    {
                        int directionTemp = ptrO[y * cols + x];
                        if (directionTemp == direction0 || directionTemp == direction1 || directionTemp == direction2)//找到符合角度阈值的，立即返回符合条件的(x,y)
                        {
                            xSeed = x;
                            ySeed = y;
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

PCLSD::PCLSD(const PCLSDParams &params) : params(params) {
}

Segments PCLSD::detect(const cv::Mat &image) {
  processImage(image);
  // std::cout << "PCLSD detected: " << segments.size() << " segments" << std::endl;
  return segments;
}

SalientSegments PCLSD::detectSalient(const cv::Mat &image) {
  processImage(image);
  // std::cout << "PCLSE detected: " << salientSegments.size() << " salient segments" << std::endl;
  return salientSegments;
}

ImageEdges PCLSD::detectEdges(const cv::Mat &image) {
  processImage(image);
  return getSegmentEdges();
}

const LineDetectionExtraInfo &PCLSD::getImgInfo() const {
  return *imgInfo;
}

void PCLSD::processImage(const cv::Mat &_image) {
  // Check that the image is a grayscale image
  cv::Mat image;
  switch (_image.channels()) {
    case 3:
      cv::cvtColor(_image, image, cv::COLOR_BGR2GRAY);
      break;
    case 4:
      cv::cvtColor(_image, image, cv::COLOR_BGRA2GRAY);
      break;
    default:
      image = _image;
      break;
  }
  assert(image.channels() == 1);
  // Clear previous state
  this->clear();

  if (image.empty()) {
    return;
  }

  // Set the global image
  // Filter the image
  if (params.ksize > 2) {
    cv::GaussianBlur(image, blurredImg, cv::Size(params.ksize, params.ksize), params.sigma);
  } else {
    blurredImg = image;
  }

  // Compute the input image derivatives
  cv::Mat edgeMap;//初始Canny图
  imgInfo = computeGradients(blurredImg, params.gradientThreshold,edgeMap);
  /*cv::imshow("mask", edgeMap);
  cv::waitKey();*/
  optImg(edgeMap, imgInfo->gImgWO);

  bool anchoThIsZero;
  uint8_t anchorTh = params.anchorThreshold;
  do {
    anchoThIsZero = anchorTh == 0;
    // Detect edges and segment in the input image
    computeAnchorPoints(imgInfo->dirImg,
                        maskImage,
                        imgInfo->gImg,
                        params.scanIntervals,
                        anchorTh,
                        anchors);

    // If we couldn't find any anchor, decrease the anchor threshold
    if (anchors.empty()) {
      // std::cout << "Cannot find any anchor with AnchorTh = " << int(anchorTh)
      //      << ", repeating process with AnchorTh = " << (anchorTh / 2) << std::endl;
      anchorTh /= 2;
    }

  } while (anchors.empty() && !anchoThIsZero);
  // LOGD << "Detected " << anchors.size() << " anchor points ";
  //drawPoint(_image);
  edgeImg = cv::Mat::zeros(imgInfo->imageHeight, imgInfo->imageWidth, CV_8UC1);
  drawer = std::make_shared<EdgeDrawer>(imgInfo,
                                        edgeImg,
                                        params.lineFitErrThreshold,
                                        params.pxToSegmentDistTh,
                                        params.minLineLen,
                                        params.treatJunctions,
                                        params.listJunctionSizes,
                                        params.junctionEigenvalsTh,
                                        params.junctionAngleTh);

  drawAnchorPoints(imgInfo->dirImg.ptr(), anchors, edgeImg.ptr());
}

LineDetectionExtraInfoPtr PCLSD::computeGradients(const cv::Mat &srcImg, short gradientTh, cv::Mat& edgeMap) {
    LineDetectionExtraInfoPtr dstInfo = std::make_shared<LineDetectionExtraInfo>();
    cv::Sobel(srcImg, dstInfo->dxImg, CV_16SC1, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);//gx
    cv::Sobel(srcImg, dstInfo->dyImg, CV_16SC1, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);//gy

    int nRows = srcImg.rows;
    int nCols = srcImg.cols;
    int i, j, m, n;
    int grayLevels = 255;
    float gNoise = 1.3333;  //1.3333 
    float thGradientLow = gNoise;

    //meaningful Length
    int thMeaningfulLength = int(2.0 * log((float)nRows * nCols) / log(8.0) + 0.5);
    float thAngle = 2 * atan(2.0 / float(thMeaningfulLength));
    //std::cout << thMeaningfulLength << std::endl;

    dstInfo->imageWidth = srcImg.cols;
    dstInfo->imageHeight = srcImg.rows;
    dstInfo->gImg = cv::Mat(srcImg.size(), dstInfo->dxImg.type());
    dstInfo->gImgWO = cv::Mat(srcImg.size(), dstInfo->dxImg.type());
    dstInfo->dirImg = cv::Mat(srcImg.size(), CV_8UC1);
    dstInfo->oriImg = cv::Mat(srcImg.size(), CV_8UC1);
    //cv::Mat gradientMap = cv::Mat::zeros(nRows, nCols, CV_32FC1);

    //calculate gradient and orientation
    int totalNum = 0;
    int times = 8;
    double anglePer = CV_PI / 8.0;
    std::vector<int> histogram(times * grayLevels, 0); 
    int16_t abs_dx, abs_dy, sum;
    for (i = 0; i < nRows; ++i)
    {
        //float* ptrG = gradientMap.ptr<float>(i);
        int16_t* ptrX = dstInfo->dxImg.ptr<int16_t>(i);
        int16_t* ptrY = dstInfo->dyImg.ptr<int16_t>(i);
        auto* pGr = dstInfo->gImg.ptr<int16_t>(i);
        auto* pGrWO = dstInfo->gImgWO.ptr<int16_t>(i);
        auto* pDir = dstInfo->dirImg.ptr<uchar>(i);
        auto* pOri = dstInfo->oriImg.ptr<uchar>(i);
        for (j = 0; j < nCols; ++j)
        {
            abs_dx = UPM_ABS(ptrX[j]);
            abs_dy = UPM_ABS(ptrY[j]); 
            sum = abs_dx + abs_dy;
            int idx= int((atan2(ptrX[j], -ptrY[j]) + CV_PI) / anglePer);  //水平角,与梯度角垂直
            pOri[j] = idx;  
            if (idx == 16)
            {
                pOri[j] = 0;
            }
            pGr[j] = sum;
            pGrWO[j] = sum;
            pDir[j] = abs_dx >= abs_dy ? UPM_EDGE_VERTICAL : UPM_EDGE_HORIZONTAL;
            //ptrG[j] = sum;
            if (pGr[j] > thGradientLow) //此时thGradientLow为1.3333
            {
                histogram[int(pGr[j] + 0.5)]++;
                totalNum++;
            }
            else
                pGr[j] = 0;
        }
    }
        //gradient statistic
    float N2 = 0;
    for (i = 0; i < histogram.size(); ++i)  //计算N_p
    {
        if (histogram[i])
            N2 += (float)histogram[i] * (histogram[i] - 1);
    }
    float pMax = 1.0 / exp((log(N2) / thMeaningfulLength));  //算的h(g_max)
    float pMin = 1.0 / exp((log(N2) / sqrt((float)nCols * nRows)));//算的h(g_min) ，原文中的N为高、宽中较大的那个，这里用 sqrt(高*宽)表示

    std::vector<float> greaterThan(times * grayLevels, 0);
    int count = 0;
    for (i = times * grayLevels - 1; i >= 0; --i)
    {
        count += histogram[i];
        float probabilityGreater = float(count) / float(totalNum);
        greaterThan[i] = probabilityGreater; //出现梯度值>i的概率
    }
    count = 0;

    //get two gradient thresholds
    int thGradientHigh = 0;
    for (i = times * grayLevels - 1; i >= 0; --i)
    {
        if (greaterThan[i] > pMax)
        {
            thGradientHigh = i;
            break;
        }
    }
    for (i = times * grayLevels - 1; i >= 0; --i)
    {
        if (greaterThan[i] > pMin)
        {
            thGradientLow = i;
            break;
        }
    }
    if (thGradientLow < gNoise) thGradientLow = gNoise;

    int VMGradient = 70;
    thGradientHigh = sqrt(thGradientHigh * VMGradient);


    //自适应阈值
    for (i = 0; i < nRows; ++i) {
        auto* pGr = dstInfo->gImg.ptr<int16_t>(i);
        for (j = 0; j < nCols; ++j) {
            if (pGr[j] <= thGradientLow) pGr[j] = 0;
        }
    }


    //原文的预定义阈值
   /* for (i = 0; i < nRows; ++i) {
        auto* pGr = dstInfo->gImg.ptr<int16_t>(i);
        for (j = 0; j < nCols; ++j) {
            if (pGr[j] <= 30) pGr[j] = 0;
        }
    }*/
        
   // std::cout << thGradientLow << "  " << thGradientHigh << std::endl;
    //canny初步图
    cv::Canny(srcImg, edgeMap, thGradientLow, thGradientHigh, 3);
   //cv::imshow("canny", dstInfo->maskImg);
   // cv::waitKey(0)
    
 return dstInfo;
}

void PCLSD::optImg(cv::Mat& edgeMap,cv::Mat& gradientMap) {
    int nRows = edgeMap.rows;
    int nCols = edgeMap.cols;
    std::vector<Pixel> gradientPoints; //坐标点
    std::vector<double> gradientValue;
    uchar* ptrE = (uchar*)edgeMap.data;
    maskImage = cv::Mat::zeros(nRows, nCols, CV_8UC1);
    uchar* ptrM =(uchar*) maskImage.data;
    auto* ptrG = gradientMap.data;
    for (int i = 0; i < nRows; ++i)
    {
        for (int j = 0; j < nCols; ++j)
        {
            if (*ptrE++)  //边缘图上显示了，则存储该像素点的坐标和梯度幅值
            {
                *ptrM++ = 1;  //在标记图像中，标记为有意义的点
                gradientPoints.emplace_back(j, i);//坐标
                gradientValue.emplace_back(ptrG[i * nCols + j]);//幅值
            }
            else
            {
                *ptrM++;
            }
        }
    }

    ptrM =(uchar*) maskImage.data;

    int numGradientPoints = gradientPoints.size();
    SortDescent(&gradientValue[0], 0, numGradientPoints - 1, &gradientPoints[0]);

    int thMeaningfulLength = int(2.0 * log((double)nRows * nCols) / log(8.0) + 0.5);
    std::vector<std::vector<Pixel>> edgeChain;
    for (int i = 0; i < numGradientPoints; ++i)
    {
        std::vector<Pixel> chain;

        int x = gradientPoints[i].x;
        int y = gradientPoints[i].y;
        do
        {
            chain.emplace_back(x, y);
            ptrM[y * nCols + x] = 0;//标记为无用的点
        } while (next(x, y));

        Pixel temp;
        for (int m = 0, n = chain.size() - 1; m < n; ++m, --n)//链翻转
        {
            temp = chain[m];
            chain[m] = chain[n];
            chain[n] = temp;
        }

        // Find and add feature pixels to the begin of the string.
        x = gradientPoints[i].x;
        y = gradientPoints[i].y;
        if (next(x, y))
        {
            do
            {
                chain.emplace_back(x, y);
                ptrM[y * nCols + x] = 0;
            } while (next(x, y));
        }

        if (chain.size() > thMeaningfulLength)
        {
            edgeChain.emplace_back(chain);
        }
    }
    //std::cout << edgeChain.size() << std::endl;
    maskImage.release();
    maskImage = cv::Mat::zeros(nRows, nCols, CV_8UC1);
    uchar* pt = (uchar*)maskImage.data;
    for (int i = 0; i < edgeChain.size(); ++i)
    {
        for (int j = 0; j < edgeChain[i].size(); ++j)
        {
            int loc = edgeChain[i][j].y * nCols + edgeChain[i][j].x;
            pt[loc] = 255;
        }
    }
 //   cv::imwrite("CannyPF.png", maskImage);
   // cv::imshow("CannyPF", maskImage);
   // cv::waitKey();
}

inline void PCLSD::computeAnchorPoints(const cv::Mat &dirImage,
                                       cv::Mat &maskImage,
                                       cv::Mat &gradImage,
                                       int scanInterval,
                                       int anchorThresh,
                                       std::vector<Pixel> &anchorPoints) {  // NOLINT

    int imageWidth = maskImage.cols;
    int imageHeight = maskImage.rows;

    // Get pointers to the thresholded gradient image and to the direction image
    auto* gradImg = gradImage.ptr<int16_t>();
    const auto* dirImg = dirImage.ptr<uint8_t>();
    auto* ptrMask = maskImage.data;

    int indexInArray;
    int w, h;
    for (w = 1; w < imageWidth - 1; w += scanInterval) {
        for (h = 1; h < imageHeight - 1; h += scanInterval) {
            indexInArray = h * imageWidth + w;

          //  if (gradImg[indexInArray] == 0) continue;   //原文的锚点选取

            // Canny用来解耦 锚点的选取与候选像素集
            if (ptrMask[indexInArray]&& gradImg[indexInArray] != 0) {// gradImg[indexInArray]!=0
                if (dirImg[indexInArray] == UPM_EDGE_HORIZONTAL) {
                    // Check if (w, h) is accepted as an anchor using the Anchor Threshold.
                    // We compare with the top and bottom pixel gradients
                    if (gradImg[indexInArray] >= gradImg[indexInArray - imageWidth] + anchorThresh &&
                        gradImg[indexInArray] >= gradImg[indexInArray + imageWidth] + anchorThresh) {
                        //Pixel p(w, h);
                        anchorPoints.emplace_back(w,h);
                    }
                }
                else {
                    // Check if (w, h) is accepted as an anchor using the Anchor Threshold.
                    // We compare with the left and right pixel gradients
                    if (gradImg[indexInArray] >= gradImg[indexInArray - 1] + anchorThresh &&
                        gradImg[indexInArray] >= gradImg[indexInArray + 1] + anchorThresh) {
                        //Pixel p(w, h);
                        anchorPoints.emplace_back(w,h);
                    }
                }
            }
        }
    }
}

void PCLSD::clear() {
  imgInfo = nullptr;
  edges.clear();
  segments.clear();
  salientSegments.clear();
  anchors.clear();
  drawer = nullptr;
  blurredImg = cv::Mat();
  edgeImg = cv::Mat();
}

inline int calculateNumPtsToTrim(int nPoints) {
  return std::min(5.0, nPoints * 0.1);
}

// Linear interpolation. s is the starting value, e the ending value
// and t the point offset between e and s in range [0, 1]
inline float lerp(float s, float e, float t) { return s + (e - s) * t; }

// Bi-linear interpolation of point (tx, ty) in the cell with corner values [[c00, c01], [c10, c11]]
inline float blerp(float c00, float c10, float c01, float c11, float tx, float ty) {
  return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

void PCLSD::drawAnchorPoints(const uint8_t *dirImg,
                             const std::vector<Pixel> &anchorPoints,
                             uint8_t *pEdgeImg) {
  assert(imgInfo && imgInfo->imageWidth > 0 && imgInfo->imageHeight > 0);
  assert(!imgInfo->gImg.empty() && !imgInfo->dirImg.empty() && pEdgeImg);
  assert(drawer);
  assert(!edgeImg.empty());

  int imageWidth = imgInfo->imageWidth;
  int imageHeight = imgInfo->imageHeight;
  bool expandHorizontally;
  int indexInArray;
  unsigned char lastDirection;  // up = 1, right = 2, down = 3, left = 4;

  if (anchorPoints.empty()) {
    // No anchor points detected in the image
    return;
  }

  const double validationTh = params.validationTh;

  for (const auto &anchorPoint: anchorPoints) {
    // LOGD << "Managing new Anchor point: " << anchorPoint;
    indexInArray = anchorPoint.y * imageWidth + anchorPoint.x;

    if (pEdgeImg[indexInArray]) {
      // If anchor i is already been an edge pixel
      continue;
    }

    // If the direction of this pixel is horizontal, then go left and right.
    expandHorizontally = dirImg[indexInArray] == UPM_EDGE_HORIZONTAL;

    /****************** First side Expanding (Right or Down) ***************/
    // Select the first side towards we want to move. If the gradient points
    // horizontally select the right direction and otherwise the down direction.
    lastDirection = expandHorizontally ? UPM_RIGHT : UPM_DOWN;

    drawer->drawEdgeInBothDirections(lastDirection, anchorPoint);
  }

  double theta, angle;
  float saliency;
  bool valid;
  int endpointDist, nOriInliers, nOriOutliers;
#ifdef UPM_SD_USE_REPROJECTION
  cv::Point2f p;
  float lerp_dx, lerp_dy;
  int x0, y0, x1, y1;
#endif
  int16_t *pDx = imgInfo->dxImg.ptr<int16_t>();
  int16_t *pDy = imgInfo->dyImg.ptr<int16_t>();
  segments.reserve(drawer->getDetectedFullSegments().size());
  salientSegments.reserve(drawer->getDetectedFullSegments().size());

  for (const FullSegmentInfo &detectedSeg: drawer->getDetectedFullSegments()) {

    valid = true;
    if (params.validate) {
      if (detectedSeg.getNumOfPixels() < 2) {
        valid = false;
      } else {
        // Get the segment angle
        Segment s = detectedSeg.getEndpoints();
        theta = segAngle(s) + M_PI_2;
        // Force theta to be in range [0, M_PI)
        while (theta < 0) theta += M_PI;
        while (theta >= M_PI) theta -= M_PI;

        // Calculate the line equation as the cross product os the endpoints
        cv::Vec3f l = cv::Vec3f(s[0], s[1], 1).cross(cv::Vec3f(s[2], s[3], 1));
        // Normalize the line direction
        l /= std::sqrt(l[0] * l[0] + l[1] * l[1]);
        cv::Point2f perpDir(l[0], l[1]);

        // For each pixel in the segment compute its angle
        int nPixelsToTrim = calculateNumPtsToTrim(detectedSeg.getNumOfPixels());

        Pixel firstPx = detectedSeg.getFirstPixel();
        Pixel lastPx = detectedSeg.getLastPixel();

        nOriInliers = 0;
        nOriOutliers = 0;

        for (auto px: detectedSeg) {

          // If the point is not an inlier avoid it
          if (edgeImg.at<uint8_t>(px.y, px.x) != UPM_ED_SEGMENT_INLIER_PX) {
            continue;
          }

          endpointDist = detectedSeg.horizontal() ?
                         std::min(std::abs(px.x - lastPx.x), std::abs(px.x - firstPx.x)) :
                         std::min(std::abs(px.y - lastPx.y), std::abs(px.y - firstPx.y));

          if (endpointDist < nPixelsToTrim) {
            continue;
          }

#ifdef UPM_SD_USE_REPROJECTION
          // Re-project the point into the segment. To do this, we should move pixel.dot(l)
          // units (the distance between the pixel and the segment) in the direction
          // perpendicular to the segment (perpDir).
          p = cv::Point2f(px.x, px.y) - perpDir * cv::Vec3f(px.x, px.y, 1).dot(l);
          // Get the values around the point p to do the bi-linear interpolation
          x0 = p.x < 0 ? 0 : p.x;
          if (x0 >= imageWidth) x0 = imageWidth - 1;
          y0 = p.y < 0 ? 0 : p.y;
          if (y0 >= imageHeight) y0 = imageHeight - 1;
          x1 = p.x + 1;
          if (x1 >= imageWidth) x1 = imageWidth - 1;
          y1 = p.y + 1;
          if (y1 >= imageHeight) y1 = imageHeight - 1;
          //Bi-linear interpolation of Dx and Dy
          lerp_dx = blerp(pDx[y0 * imageWidth + x0], pDx[y0 * imageWidth + x1],
                          pDx[y1 * imageWidth + x0], pDx[y1 * imageWidth + x1],
                          p.x - int(p.x), p.y - int(p.y));
          lerp_dy = blerp(pDy[y0 * imageWidth + x0], pDy[y0 * imageWidth + x1],
                          pDy[y1 * imageWidth + x0], pDy[y1 * imageWidth + x1],
                          p.x - int(p.x), p.y - int(p.y));
          // Get the gradient angle
          angle = std::atan2(lerp_dy, lerp_dx);
#else
          indexInArray = px.y * imageWidth + px.x;
          angle = std::atan2(pDy[indexInArray], pDx[indexInArray]);
#endif
          // Force theta to be in range [0, M_PI)
          if (angle < 0) angle += M_PI;
          if (angle >= M_PI) angle -= M_PI;
          circularDist(theta, angle, M_PI) > validationTh ? nOriOutliers++ : nOriInliers++;
        }

        valid = nOriInliers > nOriOutliers;
        saliency = nOriInliers;
      }
    } else {
      saliency = segLength(detectedSeg.getEndpoints());
    }
    if (valid) {
      const Segment &endpoints = detectedSeg.getEndpoints();
      segments.push_back(endpoints);
      salientSegments.emplace_back(endpoints, saliency);
    }
  }
}

ImageEdges PCLSD::getAllEdges() const {
  assert(drawer);
  ImageEdges result;
  for (const FullSegmentInfo &s: drawer->getDetectedFullSegments())
    result.push_back(s.getPixels());
  return result;
}

ImageEdges PCLSD::getSegmentEdges() const {
  assert(drawer);
  ImageEdges result;
  for (const FullSegmentInfo &s: drawer->getDetectedFullSegments())
    result.push_back(s.getPixels());
  return result;
}

const LineDetectionExtraInfoPtr &PCLSD::getImgInfoPtr() const {
  return imgInfo;
}

}  // namespace upm

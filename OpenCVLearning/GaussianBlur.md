`cv::GaussianBlur` 是 OpenCV 中用于对图像进行高斯模糊的重要函数。高斯模糊是一种常用的图像平滑和降噪技术，广泛应用于图像处理和计算机视觉领域。

### 函数原型

```cpp
void cv::GaussianBlur(
    InputArray src,               // 输入图像
    OutputArray dst,              // 输出图像
    Size ksize,                   // 高斯核大小
    double sigmaX,                // X方向标准差
    double sigmaY = 0,            // Y方向标准差（可选）
    int borderType = BORDER_DEFAULT  // 边界处理方式
)
```

### 参数详解

1. **输入图像 (src)**
   - 可以是单通道或多通道图像
   - 数据类型通常为 8 位无符号整数 (CV_8U)

2. **输出图像 (dst)**
   - 与输入图像大小和类型相同
   - 存储模糊处理后的结果

3. **高斯核大小 (ksize)**
   - 使用 `cv::Size(width, height)` 指定
   - 必须为正奇数，如 `Size(3,3)`, `Size(5,5)`
   - 核大小越大，模糊效果越明显

4. **标准差 (sigmaX, sigmaY)**
   - 控制高斯分布的spread
   - sigmaX：X方向标准差
   - sigmaY：Y方向标准差
   - 如果 sigmaY 为 0，则自动计算

5. **边界处理方式**
   - `BORDER_DEFAULT`：默认边界处理
   - 其他可选值：`BORDER_CONSTANT`, `BORDER_REPLICATE` 等

### 使用示例

```cpp
#include <opencv2/opencv.hpp>

int main() {
    // 读取原始图像
    cv::Mat image = cv::imread("input.jpg");
    cv::Mat blurredImage;

    // 基本高斯模糊
    cv::GaussianBlur(image, blurredImage, cv::Size(5, 5), 0);

    // 不同核大小的高斯模糊
    cv::Mat smallBlur, largeBlur;
    cv::GaussianBlur(image, smallBlur, cv::Size(3, 3), 0);
    cv::GaussianBlur(image, largeBlur, cv::Size(11, 11), 0);

    // 自定义标准差
    cv::Mat customBlur;
    cv::GaussianBlur(image, customBlur, cv::Size(5, 5), 1.5, 1.5);

    // 显示结果
    cv::imshow("Original", image);
    cv::imshow("Blurred", blurredImage);
    cv::waitKey(0);

    return 0;
}
```

### 高级用法

1. **降噪处理**
```cpp
cv::Mat denoisedImage;
cv::GaussianBlur(noisyImage, denoisedImage, cv::Size(5, 5), 1.0);
```

2. **边缘检测预处理**
```cpp
cv::Mat blurredImage, edges;
cv::GaussianBlur(image, blurredImage, cv::Size(3, 3), 0);
cv::Canny(blurredImage, edges, 100, 200);
```

3. **不同通道分别模糊**
```cpp
std::vector<cv::Mat> channels;
cv::split(image, channels);
for (auto& channel : channels) {
    cv::GaussianBlur(channel, channel, cv::Size(5, 5), 0);
}
cv::merge(channels, blurredImage);
```

### 性能和注意事项

1. 计算复杂度
   - 核大小越大，计算开销越大
   - 建议根据实际需求选择合适的核大小

2. 参数选择技巧
   - 核大小通常为奇数
   - 标准差建议在 0.3 * ((核大小 - 1) * 0.5 - 1) + 0.8 左右

3. 应用场景
   - 图像降噪
   - 图像平滑
   - 边缘检测预处理
   - 特征提取前的预处理

### 替代方案

1. `cv::blur()`：均值滤波
2. `cv::medianBlur()`：中值滤波
3. `cv::bilateralFilter()`：双边滤波（保边缘）

### 常见问题

1. 核大小必须为奇数
2. 标准差设为 0 时，将自动计算
3. 大核会显著降低处理速度

### 推荐实践

1. 选择适当的核大小
2. 根据具体应用调整标准差
3. 结合其他图像处理技术使用

`cv::GaussianBlur` 是图像处理中非常实用的函数，掌握其使用方法和参数调整对于图像预处理至关重要。
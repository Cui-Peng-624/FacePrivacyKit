#include "opencv2/opencv.hpp"

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <thread>
#include <windows.h>
#include <shobjidl.h>
#include <objbase.h> // COM 基础头文件
#include <comdef.h>  // COM 定义
#include <shlobj.h>  // Shell 对象
#include <commdlg.h>

const std::map<std::string, int> str2backend{
    {"opencv", cv::dnn::DNN_BACKEND_OPENCV}, {"cuda", cv::dnn::DNN_BACKEND_CUDA}, {"timvx", cv::dnn::DNN_BACKEND_TIMVX}, {"cann", cv::dnn::DNN_BACKEND_CANN}};

const std::map<std::string, int> str2target{
    {"cpu", cv::dnn::DNN_TARGET_CPU}, {"cuda", cv::dnn::DNN_TARGET_CUDA}, {"npu", cv::dnn::DNN_TARGET_NPU}, {"cuda_fp16", cv::dnn::DNN_TARGET_CUDA_FP16}};

class YuNet
{
public:
    YuNet(const std::string &model_path,
          const cv::Size &input_size = cv::Size(320, 320), // 设置输入图像的宽度和高度。模型需要固定大小的输入，因此这里指定预处理后图像的分辨率。
          float conf_threshold = 0.6f,                     // 置信度阈值。模型只会返回置信度高于该值的人脸检测结果。
          float nms_threshold = 0.3f,                      // 非极大值抑制（NMS）的阈值。用于消除重叠过多的检测框。用于确保检测框之间的独立性。
          int top_k = 5000,                                // 在非极大值抑制（NMS）前保留的检测框的最大数量。用于减少后处理的计算量，从而加速推理。
          int backend_id = 0,                              // 指定深度学习模型的推理后端。
          int target_id = 0)                               // 指定推理目标设备。可指定CPU，GPU，TPU
        : model_path_(model_path), input_size_(input_size),
          conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
          top_k_(top_k), backend_id_(backend_id), target_id_(target_id)
    {
        model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
    }

    /* Overwrite the input size when creating the model. Size format: [Width, Height].
     */
    void setInputSize(const cv::Size &input_size) // 设置输入图像的大小。
    {
        input_size_ = input_size;
        model->setInputSize(input_size_);
    }

    cv::Mat infer(const cv::Mat image)
    {
        cv::Mat res;
        model->detect(image, res); // 调用 model->detect(image, res) 进行人脸检测。
        return res;                // 返回一个 cv::Mat 矩阵，包含检测到的人脸数据。返回的 res 矩阵包含每个检测框的详细信息，包括位置、大小、置信度及关键点坐标。检测结果矩阵格式: (每一行表示一个检测框) - 每行包含 15 个值：[x, y, w, h, 5个关键点的x坐标, 5个关键点的y坐标, confidence] - 根据下面的代码这里是错误的，应该是14个值
    }

private:
    cv::Ptr<cv::FaceDetectorYN> model;

    std::string model_path_;
    cv::Size input_size_;
    float conf_threshold_;
    float nms_threshold_;
    int top_k_;
    int backend_id_;
    int target_id_;
};

cv::Mat visualize(const cv::Mat &image, const cv::Mat &faces, float fps = -1.f)
/*
image：输入的原始图像，是一个cv::Mat类型。表示待处理的图像，在其上绘制检测结果（人脸框、关键点等）。
faces：包含人脸检测结果的矩阵，是一个cv::Mat类型。每一行表示一个检测到的人脸，包含边界框和关键点坐标等信息。
fps：每秒帧数（FPS，Frames Per Second）。默认为-1，表示没有设置FPS。如果设置了这个参数，它会显示在图像上，作为当前检测的处理速度。
*/
{
    static cv::Scalar box_color{0, 255, 0}; // 定义人脸检测框的颜色，绿色
    static std::vector<cv::Scalar> landmark_color{
        // 定义人脸关键点颜色，右眼、左眼、鼻尖、右嘴角、左嘴角
        cv::Scalar(255, 0, 0),   // right eye - 红色
        cv::Scalar(0, 0, 255),   // left eye - 蓝色
        cv::Scalar(0, 255, 0),   // nose tip - 绿色
        cv::Scalar(255, 0, 255), // right mouth corner - 紫色
        cv::Scalar(0, 255, 255)  // left mouth corner - 青色
    };
    static cv::Scalar text_color{0, 255, 0}; // 文字颜色，用于绘制每个人脸的置信度。绿色

    auto output_image = image.clone(); // 克隆输入图像，以确保原始图像不被修改。后续在副本上修改，不会影响原始图像。

    if (fps >= 0) // 如果 fps 大于等于 0，则绘制 FPS 信息。
    {
        cv::putText(output_image, cv::format("FPS: %.2f", fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
    }

    for (int i = 0; i < faces.rows; ++i)
    {
        // Draw bounding boxes
        int x1 = static_cast<int>(faces.at<float>(i, 0));                  // 获取第 i 个人脸（人脸框）的左上角 x 坐标。
        int y1 = static_cast<int>(faces.at<float>(i, 1));                  // 获取第 i 个人脸（人脸框）的左上角 y 坐标。
        int w = static_cast<int>(faces.at<float>(i, 2));                   // 获取第 i 个人脸（人脸框）的宽度。
        int h = static_cast<int>(faces.at<float>(i, 3));                   // 获取第 i 个人脸（人脸框）的高度。
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2); // 在输出图像上绘制人脸检测框。

        // Confidence as text
        float conf = faces.at<float>(i, 14);                                                                                   // faces.at<float>(i, 14) 用于从 faces 矩阵中提取第 i 行第 14 列的数据
        cv::putText(output_image, cv::format("%.4f", conf), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color); // 使用cv::putText()函数在矩形框旁边绘制置信度的数值。

        // Draw landmarks
        for (int j = 0; j < landmark_color.size(); ++j) // landmark_color.size() 等于5，表示有5个关键点
        {
            int x = static_cast<int>(faces.at<float>(i, 2 * j + 4)), y = static_cast<int>(faces.at<float>(i, 2 * j + 5)); // 获取每个关键点的x坐标和y坐标
            cv::circle(output_image, cv::Point(x, y), 2, landmark_color[j], 2);                                           // 在输出图像上绘制每个关键点。output_image是cv::Mat类型，表示输出图像。cv::Point(x, y)是关键点的坐标。2是关键点的半径，landmark_color[j]是关键点的颜色，2是线条的粗细。
        }
    }
    return output_image;
}

std::string openFileDialog()
{
    char filename[MAX_PATH];

    OPENFILENAMEA ofn;
    ZeroMemory(&filename, sizeof(filename));
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "Image Files\0*.png;*.jpg;*.jpeg;*.bmp\0All Files\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "选择遮罩图片";
    ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

    if (GetOpenFileNameA(&ofn))
    {
        return std::string(filename);
    }
    return "";
}

// 在main函数开始前添加参数验证函数
bool validateMode(const std::string& mode) {
    return mode == "blur" || mode == "pixel" || mode == "mask";
}

int getModeFromString(const std::string& mode) {
    if (mode == "blur") return 0;
    if (mode == "pixel") return 1;
    if (mode == "mask") return 2;
    return 0; // 默认返回blur模式
}

int main(int argc, char **argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{help h           |                                   | Print this message}"
        "{input i          |                                   | Set input to a certain image}"
        "{model m          | face_detection_yunet_2023mar.onnx | Set path to the model}"
        "{backend b        | opencv                            | Set DNN backend}"
        "{target t         | cpu                               | Set DNN target}"
        "{mode            | blur                              | Set initial mode (blur/pixel/mask)}"
        "{blur_size       | 31                                | Set initial blur kernel size}"
        "{pixel_size      | 10                                | Set initial pixel size}"
        "{mask_image      | mask.png                          | Set path to mask image}"
        "{device          | 0                                 | Set camera device ID}"
        "{conf_threshold  | 0.9                               | Set face detection confidence threshold}"
        "{nms_threshold   | 0.3                               | Set NMS threshold}"
        "{top_k           | 5000                              | Set top k bounding boxes}"
        "{save s          | false                             | Save results}"
        "{vis v           | true                              | Show results}");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // 获取新增的命令行参数
    std::string mode_str = parser.get<std::string>("mode");
    int blur_size = parser.get<int>("blur_size");
    int pixel_size = parser.get<int>("pixel_size");
    std::string mask_image_path = parser.get<std::string>("mask_image");
    int device_id = parser.get<int>("device");

    // 验证模式参数
    if (!validateMode(mode_str)) {
        std::cout << "Invalid mode. Using default mode (blur)." << std::endl;
        mode_str = "blur";
    }

    //
    std::string input_path = parser.get<std::string>("input");
    std::string model_path = parser.get<std::string>("model");
    std::string backend = parser.get<std::string>("backend");
    std::string target = parser.get<std::string>("target");
    bool save_flag = parser.has("save");
    bool vis_flag = parser.has("vis");

    // model params
    float conf_threshold = parser.get<float>("conf_threshold");
    float nms_threshold = parser.get<float>("nms_threshold");
    int top_k = parser.get<int>("top_k");
    const int backend_id = str2backend.at(backend);
    const int target_id = str2target.at(target);

    // Instantiate YuNet
    YuNet model(model_path, cv::Size(320, 320), conf_threshold, nms_threshold, top_k, backend_id, target_id);

    // If input is an image
    if (!input_path.empty()) // 如果 input_path 不为空，则表示输入是图像。
    {
        auto image = cv::imread(input_path);

        // Inference
        model.setInputSize(image.size());
        auto faces = model.infer(image);

        // Print faces
        std::cout << cv::format("%d faces detected:\n", faces.rows);
        for (int i = 0; i < faces.rows; ++i)
        {
            int x1 = static_cast<int>(faces.at<float>(i, 0));
            int y1 = static_cast<int>(faces.at<float>(i, 1));
            int w = static_cast<int>(faces.at<float>(i, 2));
            int h = static_cast<int>(faces.at<float>(i, 3));
            float conf = faces.at<float>(i, 14);
            std::cout << cv::format("%d: x1=%d, y1=%d, w=%d, h=%d, conf=%.4f\n", i, x1, y1, w, h, conf);
        }

        // Draw reults on the input image
        if (save_flag || vis_flag)
        {
            auto res_image = visualize(image, faces);
            if (save_flag)
            {
                std::cout << "Results are saved to result.jpg\n";
                cv::imwrite("result.jpg", res_image);
            }
            if (vis_flag)
            {
                cv::namedWindow(input_path, cv::WINDOW_AUTOSIZE);
                cv::imshow(input_path, res_image);
                cv::waitKey(0);
            }
        }
    }
    else // Call default camera
    {
        auto cap = cv::VideoCapture(device_id);
        if (!cap.isOpened()) {
            std::cout << "Error: Could not open camera " << device_id << std::endl;
            return -1;
        }

        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        model.setInputSize(cv::Size(w, h));

        auto tick_meter = cv::TickMeter();
        cv::Mat frame;

        // 创建窗口和控制面板
        cv::namedWindow("YuNet Demo", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Parameters", cv::WINDOW_AUTOSIZE); // 新建一个参数控制窗口

        // 创建参数控制
        int blur_kernel_size = 31; // 高斯模糊核大小
        int pixel_size = 10;       // 像素化大小
        int effect_mode = 0;       // 0: 高斯模糊, 1: 像素化, 2: 遮罩
        const int max_kernel_size = 99;
        const int max_pixel_size = 50;

        // 创建参数调节的滑动条
        auto onTrackbar = [](int, void*) { };
        cv::createTrackbar("Blur Kernel Size", "Parameters", nullptr, max_kernel_size, onTrackbar);
        cv::setTrackbarPos("Blur Kernel Size", "Parameters", blur_kernel_size);
        
        cv::createTrackbar("Pixel Size", "Parameters", nullptr, max_pixel_size, onTrackbar);
        cv::setTrackbarPos("Pixel Size", "Parameters", pixel_size);

        // 加载默认遮罩图片
        cv::Mat mask_image;
        if (effect_mode == 2) {
            mask_image = cv::imread(mask_image_path, cv::IMREAD_UNCHANGED);
            if (mask_image.empty()) {
                std::cout << "Warning: Cannot load mask image from " << mask_image_path << std::endl;
                // 创建一个默认的彩色遮罩
                mask_image = cv::Mat(100, 100, CV_8UC3, cv::Scalar(255, 200, 200));
            } else {
                std::cout << "Successfully loaded mask image from " << mask_image_path << std::endl;
            }
        } else {
            // 为其他模式创建一个默认遮罩
            mask_image = cv::Mat(100, 100, CV_8UC3, cv::Scalar(255, 200, 200));
        }

        // 用于处理遮罩图片更新的标志和变量
        bool updating_mask = false;
        std::string new_mask_path;

        // 辅助函数
        auto getValidKernelSize = [](int size)
        {
            size = std::max(1, size);
            return size % 2 == 0 ? size + 1 : size;
        };

        auto pixelate = [](cv::Mat &roi, int pixel_size)
        {
            cv::Size small_size(
                std::max(1, roi.cols / pixel_size),
                std::max(1, roi.rows / pixel_size));
            cv::Mat temp;
            cv::resize(roi, temp, small_size, 0, 0, cv::INTER_LINEAR);
            cv::resize(temp, roi, roi.size(), 0, 0, cv::INTER_NEAREST);
        };

        // 遮罩处理函数
        auto applyMask = [](cv::Mat &roi, const cv::Mat &mask)
        {
            cv::Mat resized_mask;
            cv::resize(mask, resized_mask, roi.size());

            // 如果遮罩图片包含alpha通道
            if (resized_mask.channels() == 4)
            {
                std::vector<cv::Mat> channels;
                cv::split(resized_mask, channels);

                // 使用alpha通道作为掩码
                cv::Mat mask_alpha;
                channels[3].convertTo(mask_alpha, CV_32F, 1.0 / 255.0);

                // 合并RGB通道
                cv::Mat mask_rgb;
                std::vector<cv::Mat> rgb_channels(channels.begin(), channels.begin() + 3);
                cv::merge(rgb_channels, mask_rgb);

                // 应用alpha混合
                cv::Mat roi_f;
                roi.convertTo(roi_f, CV_32F, 1.0 / 255.0);
                mask_rgb.convertTo(mask_rgb, CV_32F, 1.0 / 255.0);

                for (int i = 0; i < roi.rows; i++)
                {
                    for (int j = 0; j < roi.cols; j++)
                    {
                        float alpha = mask_alpha.at<float>(i, j);
                        roi_f.at<cv::Vec3f>(i, j) = mask_rgb.at<cv::Vec3f>(i, j) * alpha +
                                                    roi_f.at<cv::Vec3f>(i, j) * (1 - alpha);
                    }
                }
                roi_f.convertTo(roi, CV_8UC3, 255.0);
            }
            else
            {
                // 如果没有alpha通道，直接覆盖
                resized_mask.copyTo(roi);
            }
        };

        // 设置初始参数
        effect_mode = getModeFromString(mode_str);
        blur_kernel_size = blur_size;
        pixel_size = pixel_size;

        // 如果是遮罩模式，加载指定的遮罩图片
        if (effect_mode == 2) {
            cv::Mat custom_mask = cv::imread(mask_image_path, cv::IMREAD_UNCHANGED);
            if (!custom_mask.empty()) {
                mask_image = custom_mask;
                std::cout << "已加载遮罩图片: " << mask_image_path << std::endl;
            } else {
                std::cout << "无法加载遮罩图片: " << mask_image_path << "，使用默认遮罩" << std::endl;
            }
        }

        while (true)
        {
            // 处理键盘输入
            int key = cv::waitKey(1);
            if (key == 27) // ESC键退出
                break;
            else if (key == '1') // 按1切换到高斯模糊模式
                effect_mode = 0;
            else if (key == '2') // 按2切换到像素化模式
                effect_mode = 1;
            else if (key == '3') // 按3切换到遮罩模式
                effect_mode = 2;
            else if (key == 'u' || key == 'U')
            {
                std::string input_path = openFileDialog();
                if (!input_path.empty())
                {
                    cv::Mat new_mask = cv::imread(input_path, cv::IMREAD_UNCHANGED);
                    if (!new_mask.empty())
                    {
                        mask_image = new_mask;
                        std::cout << "遮罩图片已更新: " << input_path << std::endl;
                    }
                    else
                    {
                        std::cout << "无法加载图片: " << input_path << std::endl;
                    }
                }
            }

            bool has_frame = cap.read(frame);
            if (!has_frame)
            {
                std::cout << "No frames grabbed! Exiting ...\n";
                break;
            }

            tick_meter.start();
            cv::Mat faces = model.infer(frame);
            tick_meter.stop();

            // 对每个检测到的人脸进行处理
            for (int i = 0; i < faces.rows; ++i)
            {
                int x1 = static_cast<int>(faces.at<float>(i, 0));
                int y1 = static_cast<int>(faces.at<float>(i, 1));
                int w = static_cast<int>(faces.at<float>(i, 2));
                int h = static_cast<int>(faces.at<float>(i, 3));

                // 确保人脸区域在图像范围内
                x1 = std::max(0, x1);
                y1 = std::max(0, y1);
                w = std::min(w, frame.cols - x1);
                h = std::min(h, frame.rows - y1);

                // 获取人脸区域
                cv::Rect face_rect(x1, y1, w, h);
                cv::Mat face_roi = frame(face_rect);

                // 根据选择的模式应用效果
                if (effect_mode == 0)
                { // 高斯模糊模式
                    int current_kernel_size = getValidKernelSize(blur_kernel_size);
                    cv::GaussianBlur(face_roi, face_roi,
                                     cv::Size(current_kernel_size, current_kernel_size), 0);
                }
                else if (effect_mode == 1)
                { // 像素化模式
                    pixelate(face_roi, std::max(1, pixel_size));
                }
                else
                { // 遮罩模式
                    applyMask(face_roi, mask_image);
                }
            }

            // Draw results on the input image
            auto res_image = visualize(frame, faces, (float)tick_meter.getFPS());

            // 显示操作说明和当前状态
            cv::putText(res_image,
                        "Press '1': Blur  '2': Pixelate  '3': Mask  'u': Update mask  'ESC': Exit",
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 2);

            // 更新参数显示
            std::string mode_str;
            std::string param_str;
            switch (effect_mode)
            {
            case 0:
                mode_str = "Current Mode: Gaussian Blur";
                param_str = cv::format("Parameter: Kernel Size = %d",
                                       getValidKernelSize(blur_kernel_size));
                break;
            case 1:
                mode_str = "Current Mode: Pixelate";
                param_str = cv::format("Parameter: Pixel Size = %d", pixel_size);
                break;
            case 2:
                mode_str = "Current Mode: Mask";
                param_str = "Press 'u' to update mask image";
                break;
            }

            cv::putText(res_image, mode_str,
                        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 2);
            cv::putText(res_image, param_str,
                        cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 2);

            cv::imshow("YuNet Demo", res_image);
            tick_meter.reset();
        }
    }

    return 0;
}
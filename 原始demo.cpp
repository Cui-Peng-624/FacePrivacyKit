#include "opencv2/opencv.hpp"

#include <map>
#include <vector>
#include <string>
#include <iostream>

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
    static std::vector<cv::Scalar> landmark_color{ // 定义人脸关键点颜色，右眼、左眼、鼻尖、右嘴角、左嘴角
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
        int x1 = static_cast<int>(faces.at<float>(i, 0)); // 获取第 i 个人脸（人脸框）的左上角 x 坐标。
        int y1 = static_cast<int>(faces.at<float>(i, 1)); // 获取第 i 个人脸（人脸框）的左上角 y 坐标。
        int w = static_cast<int>(faces.at<float>(i, 2)); // 获取第 i 个人脸（人脸框）的宽度。
        int h = static_cast<int>(faces.at<float>(i, 3)); // 获取第 i 个人脸（人脸框）的高度。
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2); // 在输出图像上绘制人脸检测框。

        // Confidence as text
        float conf = faces.at<float>(i, 14); // faces.at<float>(i, 14) 用于从 faces 矩阵中提取第 i 行第 14 列的数据
        cv::putText(output_image, cv::format("%.4f", conf), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color); // 使用cv::putText()函数在矩形框旁边绘制置信度的数值。

        // Draw landmarks
        for (int j = 0; j < landmark_color.size(); ++j) // landmark_color.size() 等于5，表示有5个关键点
        {
            int x = static_cast<int>(faces.at<float>(i, 2 * j + 4)), y = static_cast<int>(faces.at<float>(i, 2 * j + 5)); // 获取每个关键点的x坐标和y坐标
            cv::circle(output_image, cv::Point(x, y), 2, landmark_color[j], 2); // 在输出图像上绘制每个关键点。output_image是cv::Mat类型，表示输出图像。cv::Point(x, y)是关键点的坐标。2是关键点的半径，landmark_color[j]是关键点的颜色，2是线条的粗细。
        }
    }
    return output_image;
}

int main(int argc, char **argv)
{
    cv::CommandLineParser parser(argc, argv,
                                 "{help  h           |                                   | Print this message}" // help h: 输出帮助信息。
                                 "{input i           |                                   | Set input to a certain image, omit if using camera}" // input i: 设置输入为特定图像，如果使用摄像头则省略。  
                                 "{model m           | face_detection_yunet_2023mar.onnx | Set path to the model}" // model m: 设置模型路径。
                                 "{backend b         | opencv                            | Set DNN backend}" // backend b: 设置DNN后端。
                                 "{target t          | cpu                               | Set DNN target}" // target t: 设置DNN目标。
                                 "{save s            | false                             | Whether to save result image or not}" // save s: 是否保存结果图像。
                                 "{vis v             | false                             | Whether to visualize result image or not}" // vis v: 是否可视化结果图像。
                                 /* model params below*/
                                 "{conf_threshold    | 0.9                               | Set the minimum confidence for the model to identify a face. Filter out faces of conf < conf_threshold}" // conf_threshold: 设置模型识别人脸的最小置信度。过滤掉置信度低于 conf_threshold 的人脸。
                                 "{nms_threshold     | 0.3                               | Set the threshold to suppress overlapped boxes. Suppress boxes if IoU(box1, box2) >= nms_threshold, the one of higher score is kept.}" // nms_threshold: 设置非极大值抑制（NMS）的阈值。用于消除重叠过多的检测框。
                                 "{top_k             | 5000                              | Keep top_k bounding boxes before NMS. Set a lower value may help speed up postprocessing.}"); // top_k: 在非极大值抑制（NMS）前保留的检测框的最大数量。用于减少后处理的计算量，从而加速推理。
    if (parser.has("help")) // 如果命令行中包含 help 参数，则输出帮助信息。
    {
        parser.printMessage();
        return 0;
    }

    //
    std::string input_path = parser.get<std::string>("input");
    std::string model_path = parser.get<std::string>("model");
    std::string backend = parser.get<std::string>("backend");
    std::string target = parser.get<std::string>("target");
    bool save_flag = parser.get<bool>("save");
    bool vis_flag = parser.get<bool>("vis");

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
        int device_id = 0;
        auto cap = cv::VideoCapture(device_id); // 创建一个 VideoCapture 对象，用于从默认摄像头读取视频流。
        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)); // 获取视频流的宽度。
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)); // 获取视频流的高度。
        model.setInputSize(cv::Size(w, h)); // 设置模型输入图像的大小。

        auto tick_meter = cv::TickMeter(); // 创建一个 TickMeter 对象，用于测量时间。   
        cv::Mat frame; // 创建一个 cv::Mat 对象，用于存储每一帧图像。
        while (cv::waitKey(1) < 0) // 等待用户按键，如果用户按下任意键，则退出循环。
        {
            bool has_frame = cap.read(frame); // 从摄像头读取一帧图像，并存储在 frame 中。  
            if (!has_frame) // 如果读取失败，则输出错误信息并退出循环。
            {
                std::cout << "No frames grabbed! Exiting ...\n";
                break;
            }

            // Inference
            tick_meter.start(); // 开始测量时间。   
            cv::Mat faces = model.infer(frame); // 调用模型进行推理，返回检测到的人脸数据。
            tick_meter.stop(); // 停止测量时间。

            // Draw results on the input image
            auto res_image = visualize(frame, faces, (float)tick_meter.getFPS()); // 在输入图像上绘制检测结果，并显示FPS。
            // Visualize in a new window
            cv::imshow("YuNet Demo", res_image); // 显示结果图像。

            tick_meter.reset(); // 重置测量时间。
        }
    }

    return 0;
}
# 人脸检测与隐私保护系统

这是一个基于OpenCV和YuNet模型的实时人脸检测与隐私保护系统。该系统能够通过摄像头实时检测人脸，并提供多种隐私保护效果。

## 版本说明

本项目提供四个不同版本的实现：

1. **原始demo.cpp**: 基础版本，提供人脸检测的核心功能
2. **demo1204_晚18.cpp**: 添加基本隐私保护功能（高斯模糊和像素化）
3. **demo1204_晚19.cpp**: 完整版本，包含所有功能和GUI界面
4. **demo.cpp**: 命令行版本，支持脚本控制和批处理

详细的版本功能和使用方法请参考 [使用说明.md](使用说明.md)

## 功能特点

- 实时人脸检测
- 多种隐私保护模式：
  - 高斯模糊
  - 像素化
  - 自定义遮罩
- 可调节参数：
  - 模糊强度
  - 像素化大小
  - 自定义遮罩图片
- 实时FPS显示
- 支持图片和摄像头输入
- 提供GUI和命令行两种控制方式

## 依赖项

- OpenCV 4.x
- Windows系统（使用了Windows API进行文件选择）
- C++编译器（支持C++11及以上）

## 使用说明

### 命令行参数

```bash
用法：
  ./程序名 [参数]
参数：
  -h, --help            显示帮助信息
  -i, --input          设置输入图片路径（不设置则使用摄像头）
  -m, --model          设置模型路径（默认：face_detection_yunet_2023mar.onnx）
  -b, --backend        设置DNN后端（默认：opencv）
  -t, --target         设置DNN目标（默认：cpu）
  --conf_threshold     设置置信度阈值（默认：0.9）
  --nms_threshold      设置NMS阈值（默认：0.3）
```

### 快捷键

- `1`: 切换到高斯模糊模式
- `2`: 切换到像素化模式
- `3`: 切换到遮罩模式
- `A/D`: 调整当前模式参数
- `U`: 更新遮罩图片
- `ESC`: 退出程序

## 构建说明

1. 确保已安装OpenCV 4.x
2. 使用CMake构建项目
3. 将YuNet模型文件放在执行文件同目录下

## 注意事项

- 请确保摄像头正常工作
- 模型文件必须存在于指定路径
- 遮罩图片支持带Alpha通道的PNG格式
- 详细的版本使用说明请参考 [使用说明.md](使用说明.md)
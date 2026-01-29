from ultralytics import YOLO
import os
import torch
import shutil
import argparse
import platform

class ExportModelManager():
    def __init__(self, model_path: str, save_path: str = "/home/user/hjh", imgsz: int = 640, mode: str = "detect",
                 slice: bool = False, batch: int = 1, device: str = "cuda:0"):
        """
        初始化 YOLO 模型转换器。
        :param model_path: YOLO 模型路径 (.pt)
        :param save_path: 转换后模型的保存路径
        :param imgsz: 输入图片大小
        :param mode: 模型模式 (detect, segment, classify, pose)
        :param device: 设备类型 (cuda 或 cpu)
        """
        self.model_path = model_path
        self.save_path = os.path.join(save_path, "models")
        self.imgsz = imgsz
        self.slice = slice
        self.batch = batch
        self.mode = mode
        self.device = self._check_device(device)
        self.model = None
        # 支持的导出格式（标注兼容系统）
        self.supported_formats = {
            "coreml": "苹果生态专属（仅macOS/Linux支持）",
            "onnx": "通用跨平台格式（全系统支持）",
            "tensorrt": "NVIDIA GPU加速（需CUDA环境）",
            "tflite": "移动端/嵌入式设备（全系统支持）",
            "torchscript": "PyTorch原生格式（全系统支持）"
        }
        # 当前系统
        self.current_system = platform.system()

    def _check_device(self, device):
        """检查 CUDA 是否可用，如果不可用则切换为 CPU"""
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA 不可用，切换到 CPU 运行。")
            return "cpu"
        return device

    def _interactive_choose_format(self):
        """交互式让用户选择要转换的格式"""
        print("\n===== 支持的模型格式 =====")
        # 过滤当前系统不支持的格式
        for idx, (fmt, desc) in enumerate(self.supported_formats.items(), 1):
            # Windows 下标注 CoreML 不支持
            if self.current_system == "Windows" and fmt == "coreml":
                desc += "【当前系统不支持】"
            print(f"{idx}. {fmt}: {desc}")
        
        # 循环直到用户输入有效
        while True:
            try:
                choice = input("\n请输入要转换的格式序号（或直接输入格式名，如 onnx）：").strip()
                # 处理序号输入
                if choice.isdigit():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(self.supported_formats):
                        selected_format = list(self.supported_formats.keys())[choice_idx]
                    else:
                        print(f"请输入 1-{len(self.supported_formats)} 之间的序号！")
                        continue
                # 处理格式名输入
                else:
                    if choice in self.supported_formats:
                        selected_format = choice
                    else:
                        print(f"不支持该格式！支持的格式：{list(self.supported_formats.keys())}")
                        continue
                
                # 检查 Windows 下选择 CoreML 的情况
                if self.current_system == "Windows" and selected_format == "coreml":
                    confirm = input("警告：Windows 系统不支持 CoreML 格式，是否切换为 ONNX？(y/n)：").strip().lower()
                    if confirm == "y":
                        selected_format = "onnx"
                        print("已自动切换为 ONNX 格式！")
                    else:
                        print("取消转换，退出程序。")
                        exit(0)
                
                return selected_format
            except KeyboardInterrupt:
                print("\n用户取消操作，退出程序。")
                exit(0)
            except Exception as e:
                print(f"输入错误：{e}，请重新输入！")

    def load_model(self):
        """加载 YOLO 模型"""
        try:
            self.model = YOLO(self.model_path)
            print(f"成功加载模型: {self.model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e

    def convert(self, target_format: str = None):
        """
        转换 YOLO 模型为指定格式
        :param target_format: 目标格式（None 则进入交互模式选择）
        """
        if self.model is None:
            self.load_model()

        # 如果没指定格式，进入交互选择
        if target_format is None:
            target_format = self._interactive_choose_format()

        try:
            # 构建导出参数（通用参数）
            export_params = {
                "format": target_format,
                "dynamic": self.slice,
                "batch": self.batch,
                "imgsz": self.imgsz,
                "device": self.device,
                "simplify": True,
            }
            # 针对不同格式的特殊参数调整
            if target_format == "tensorrt":
                export_params["half"] = True  # TensorRT 开启半精度加速
            elif target_format == "tflite":
                export_params["int8"] = False  # 关闭INT8量化（避免兼容性问题）

            print(f"\n正在转换模型 {self.model_path} 为 {target_format} 格式，参数: {export_params}")
            # 执行导出
            output_path = self.model.export(**export_params)
            
            if not output_path:
                raise Exception(f"{target_format} 格式导出返回空路径，可能导出失败！")
            
            print(f"{target_format} 模型转换成功！输出路径: {output_path}")

            # 确保保存目录存在
            os.makedirs(self.save_path, exist_ok=True)
            # 复制转换后的模型到指定保存路径
            target_path = os.path.join(self.save_path, os.path.basename(output_path))
            shutil.move(output_path, target_path)
            print(f"{target_format} 模型已保存到: {target_path}")

        except Exception as e:
            print(f"转换模型 {self.model_path} 为 {target_format} 格式失败: {e}")
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 模型转换脚本（支持交互选择格式）")
    parser.add_argument("--model_path", type=str, help="YOLO 模型文件路径 (.pt)")
    parser.add_argument("--save_path", type=str, default="./models", help="转换后模型的保存路径")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--mode", type=str, default="segment", choices=["detect", "segment", "classify", "pose"],
                        help="检测模式")
    parser.add_argument("--slice", action="store_true", help="是否使用动态尺寸")
    parser.add_argument("--batch", type=int, default=1, help="批量大小")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备类型 (cuda 或 cpu)")
    parser.add_argument("--format", type=str, choices=["coreml", "onnx", "tensorrt", "tflite", "torchscript"],
                        help="目标格式（不指定则进入交互模式）")

    args = parser.parse_args()

    # 处理模型路径：如果命令行没传，交互式输入
    if not args.model_path:
        args.model_path = input("请输入 YOLO 模型文件路径 (.pt)：").strip()
        if not os.path.exists(args.model_path):
            print(f"模型文件不存在：{args.model_path}")
            exit(1)

    # 初始化转换器
    converter = ExportModelManager(
        model_path=args.model_path,
        save_path=args.save_path,
        imgsz=args.imgsz,
        mode=args.mode,
        slice=args.slice,
        batch=args.batch,
        device=args.device
    )

    # 执行转换（指定格式则直接转，否则交互选择）
    converter.convert(target_format=args.format)
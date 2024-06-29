## 实验任务
Implement [ACL](https://github.com/facebookresearch/Adversarial-Continual-Learning) under [LibContinual](https://github.com/RL-VIG/LibContinual).

## 代码环境
我所使用的`conda`环境见 [njubox](https://box.nju.edu.cn/d/1acf2a8440c94731a9ab/) 。

**Note**:与`LibContinual`与`ACL`的文档中要求有所不同，为了能在`A800`上运行改变了`torch`,`torchvision`,`cuDNN`的版本。

- Python: 3.8.0

## 数据集
`CIFAR100`存放位置: `/dataset/cifar100`，将`CIFAR100`分成20个任务，每个任务包含5个类别。

`CIFAR100` 数据集见 [Google Drive](https://drive.google.com/drive/folders/1EL46LQ3ww-F1NVTwFDPIg-nO198cUqWm) 。

## 代码运行(理论流程，未经过测验)

- **Step1**: 修改`run_trainer.py`中`Config`参数为`./config/acl.yaml`
- **Step2**：配置`./config/acl.yaml`文件中的参数
    - `data_root`: 数据集路径
    - `save_path`: 日志保存路径
- **Step3**: 运行代码`python run_trainer.py`
- **Step4**：日志保存在配置文件中`save_path`路径下

## 目标文件
Files need to be modified:
- [ ] `/core/model/acl.py`
- [ ] `/config/acl.yaml`
- [ ] `/reproduce/acl/README.md`
- [ ] `/resources/imgs/acl.png`

## 定义模型

In `acl.py`, we define the model `ACL` as follows:
- [ ] `class Model(nn.Module)`
- [ ] `class ACL(Finetune)`
    - [ ] `__init__` : 初始化函数，设置算法所需的初始化参数。
    - [ ] `observe` : 在训练阶段调用，输入一个批次的训练样本并返回预测结果、准确率和前向损失。（在训练阶段面对一个batch的数据，模型如何计算损失，如何进行参数更新）
    - [ ] `inference` : 在推理阶段调用，输入一个批次的测试样本并返回分类结果和准确率。（在测试阶段面对一个batch的数据，模型如何进行前向推理）
    - [ ] `forward` : 重写 `PyTorch` 中 `Module` 的 `forward` 函数，返回主干网络的输出。
    - [ ] `before_task` : 在每个任务开始训练之前调用，用于调整模型结构、训练参数等，需要用户自定义。（每个任务训练开始之前的自定义操作）
    - [ ] `after_task` : 在每个任务开始训练之后调用，用于调整模型结构、缓冲区等，需要用户自定义。（每个任务训练结束之后的的自定义操作）
    - [ ] `get_parameters` : 在每个任务开始训练之前调用，返回当前任务的训练参数。

## 配置文件
[LibContinualDocs](https://libcontinual.readthedocs.io/en/latest/docs/config_file_en.html)

In `acl.yaml`, we set the configuration for the model `ACL` as follows:
- [ ] Dataset：所用数据集
- [ ] Optimizer：训练的优化器与学习率调度器信息
- [ ] Backbone：配置模型骨干网络的信息
- [ ] Buffer：配置数据存储策略
- [ ] Algorithm：与方法相关的参数
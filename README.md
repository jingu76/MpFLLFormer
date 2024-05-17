# 目录结构说明

- algorithm：存放不同算法模型以及相关训练与测试脚本（每个算法模型单独建一个文件夹） 
    - Algorithm1
        - network：存放模型定义相关代码
        - train：存放训练相关代码
        - test：存放推理/测试相关代码
- dataset：存放Dataset的相关实现
- loss：存放一些自定义的Loss实现
- optimizer：存放一些自定义的优化器以及lr scheduler实现
- scripts：存放一些通用脚本，如数据集转换、统一测试脚本等
- utils：存放一些通用的工具类代码，如指标计算，后处理等
- assets：存放一些文档中使用的插图等

# 代码编写要求

- import 路径统一使用项目根目录
- 无关文件在.gitignore中进行指定
- 如果算法有特殊的依赖需求，在对应的算法目录中增加相应的requirements.txt或README.md说明依赖配置方式

# 依赖安装

先安装torch 1.13.1，参照torch官网

```
pip install -r requirements.txt
pip install 'monai[all]'
pip install monai==1.2.0
```
# 内容介绍见:
- [密集热点区域无线网络的性能分析与优化](https://jimyma.github.io/2018/03/12/UDNs/)
- [小基站服务用户数的统计特性的分析](https://jimyma.github.io/2018/03/15/statistics_select_user_num/)
- [有关超密集组网的问题汇总（持续更新）](https://jimyma.github.io/2018/03/22/UDNsQA/)


# 主要程序清单和相应程序简要说明

毕设的程序采用:
- **Python 3.5**
- **Ubuntu 16.04**
- **PyCharm 2017.3**
- **Jupyter Notebook**

该文件夹下的文件按树形结构表示如下：

```
program
├── README.md
├── setup.py
├── examples
└── udntools
```

文件类型和文件说明如下表所示：

|  文件名  |文件类型|             文件说明            |
|:---------:|:-----:|:-------------------------------:|
|README.md|   -   |存放介绍毕设内容的一个超链接|
|setup.py |   -   |注册编写的包到Python的环境变量列表里|
|udntools |   d   |超密集组网的工具包，是程序的主要部分|
|examples |   d   |用于存放应用工具包得到的理论和仿真图|

程序的总体架构为一个两级的架构：
- "udntools" 包中包含用于实现理论分析和仿真的类和方法，
- "example" 文件夹中存放通过仿真得到的仿真曲线。


# 主要程序清单和相应程序简要说明

毕设的程序采用:
- **Python 3.5** 
- **Ubuntu 16.04** 
- **PyCharm 2017.3** 
- **Jupyter Notebook**

下面对程序进行简要的说明，下面按层次介绍，第一个层次为网络的整体的架构，按树形结构表示如下：

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

下面对 "udntools" 工具包和 "examples" 文件夹中的内容做详细的介绍：


## "udntools" 工具包

"udntools" 工具包中的内容的树形结构表示如下：

```
udntools
├── bs
├── channel
├── __init__.py
├── region
├── ue
└── utils
```

其中 "\_\_init\_\_.py" 用于辅助完成包内部的相互调用和外部对包的调用。其他文件的类型和文件的说明如下表所示：

|  文件名  |文件类型|   文件说明      |
|:-------:|:-----:|:-------------:|
|    bs   |   d   |表示微基站属性的包|
|    ue   |   d   |表示用户的属性的包|
|  channel|   d   |表示信道的特性的包|
|  region |   d   |网络性能仿真的接口| 
|  utils  |   d   |辅助实现仿真的方法| 

"udntools" 工具包中包含 5 个子包，其中前 4 个子包分别用于完成对基站，用户，信道，区域的建模，
最后的 "utils" 包包含用于辅助实现建模功能的方法。

下面对这 5 个子包做详细的介绍。

### "bs" 子包

"bs"子包中的内容的树形结构表示如下：

```
bs
├── base_bs.py
└── __init__.py
```

"base_bs.py" 中为用于表征用户特性的类 **BaseBS**:

对 **BaseBS** 中的主要对象和方法进行说明：

- BaseBS
    - Inherit from class **Object**

|  名称     |  类型 |      说明      |
|:--------:|:-----:|:-------------:|
|bs_number_|  对象  | 基站的个数| 
|bs_power_ |  对象  | 基站的功率|
|bs_distribution_|对象|基站的分布|
|bs_position_| 对象 | 基站的位置|
|set_bs_to_region|抽象方法|在区域内生成基站|
|select_ue| 抽象方法 |寻找基站服务的用户集合|

**BaseBS** 继承自 **Python** 的基类 **Object**。
用于存放基站的个数、功率、分布、位置等对象。
同时根据基站的实际的物理特性，其他继承**BaseBS**的子类必须实现放置基站和用户选择基站这两个抽象方法。

### "ue"子包

"ue"子包中的内容的树形结构表示如下：

```
ue
├── base_ue.py
└── __init__.py
```

"base_ue.py" 中为用于表征用户特性的类 **BaseUE**:

对 **BaseUE** 中的主要对象和方法进行说明：

- BaseUE
    - Inherit from class **Object**
    
|  名称     |  类型 |      说明      |
|:--------:|:-----:|:-------------:|
|ue_number_|  对象  | 用户的个数| 
|ue_distribution_|对象|用户的分布|
|bs_position_| 对象 | 用户的位置|
|set_ue_to_region|抽象方法|在区域内生成用户|

**BaseUE** 继承自 **Python** 的基类 **Object**。
用于存放用户的个数、分布、位置等对象。
其他继承**BaseUE**的子类必须实现用户生成这个抽象方法。

### "Channel" 子包

"channel" 子包中的内容的树形结构表示如下：

```
channel
├── __init__.py
├── large_fade_channel.py
└── small_fade_channel.py
```

#### "large_fade_channel.py"

"large_fade_channel.py" 中为用于表征信道大尺度衰落特性的类 **LargeFadeChannel**:

对 **LargeFadeChannel** 中的主要对象和方法进行说明：

- LargeFadeChannel
    - Inherit from class **Object**

|  名称     |  类型 |      说明      |
|:--------:|:-----:|:-------------:|
|path_loss_factor_|对象|大尺度衰落系数|
|large_fade_factor_matrix|方法|发射信号经过大尺度衰落后的功率|

#### "small_fade_channel.py"

"small_fade_channel.py" 中为用于表征信道小尺度衰落特性的类 **SmallFadeChannel**:

对 **SmallFadeChannel** 中的主要对象和方法进行说明：

|  名称     |  类型 |      说明      |
|:--------:|:-----:|:-------------:|
|small_fade_|对象|小尺度衰落的类型|
|h_matrix|对象|信道系数|
|generate_h_matrix|方法|生成小尺度衰落系数矩阵|

### "Region" 子包

"region" 子包中的内容的树形结构表示如下：

```
region
├── base_region.py
├── comp_service_region.py
├── __init__.py
└── service_region.py
```

#### "base_region.py"

"base_region.py" 中为用于设定区域的基本特性的类 **BaseRegion**:

对 **BaseRegion** 中的主要对象和方法进行说明：

- BaseRegion
    - Inherit from class **Object**

|  名称     |  类型 |      说明      |
|:--------:|:-----:|:-------------:|
|x_min|对象|区域的左边界|
|x_max|对象|区域的右边界|
|y_min|对象|区域的下边界|
|y_max|对象|区域的上边界|
|ground_position_|对象|用于存放生成的格点坐标|
|get_ground|方法|得到格点坐标的矩阵|

#### "service_region.py"

"service_region.py" 为对密集热点区域无线网络进行性能分析的接口 **ServiceRegion**:

对 **ServiceRegion** 中的主要对象和方法进行说明：

- BaseRegion
    - Inherit from class **BaseRegion, BaseBS, BaseUE**

|  名称     |  类型 |      说明      |
|:--------:|:-----:|:-------------:|
|bs_ue_dict_|对象|存放基站对应其所服务的用户的字典|
|kill_ue|方法|删除区域内的用户|
|set_bs_to_region|方法|区域内基站部署|
|set_ue_to_region|方法|区域内基站部署|
|set_ue_sigma|方法|设定用户的分散程度|
|set_ue_distribution|方法|设定用户的分布|
|select_ue|方法|得到bs_ue_dict_|

#### "comp_service_region.py"

"comp_service_region.py" 为对密集热点区域无线网络进行性能优化的接口 **CompServiceRegion**:

对 **CompServiceRegion** 中的主要对象和方法进行说明：

- BaseRegion
    - Inherit from class **ServiceRegion, LargeFadeChannel, SmallFadeChannel**

|  名称     |  类型 |      说明      |
|:--------:|:-----:|:-------------:|
|cluster_set_|对象|基站分簇得到的集合|
|cluster_bs_position_|对象|簇内基站的坐标|
|cluster_ue_set_|对象|根据基站分簇结果对用户分簇|
|cluster_ue_position_|对象|簇内用户的坐标|
|self.sir_array|对象|每个用户的信干比性能|
|cluster_by_kmeans|方法|基于 Kmeans 的基站分簇算法|
|cluster_by_dfs|方法|基于深度优先搜索的基站分簇算法|
|get_cluster_ue_position|方法|得到 cluster_ue_set_ 和 cluster_ue_position_ |
|zfbf_equal_allocation|方法|基于 ZFBF 的多用户联合传输算法|
|sir_array_sim|方法|得到用户的信干比的性能|

### "utils" 子包

"utils" 子包中的内容的树形结构表示如下：

```
utils
├── ase_theory.py
├── cdf.py
├── dfs_dict_by_distance.py
├── dim2_distance.py
├── __init__.py
└── pc_theory.py
```

该包中包含辅助网络性能分析的类方法，每个文件中包含的功能如下：

|  文件名     |  文件类型 |      说明      |
|:----------:|:--------:|:-------------:|
|ase_theory.py|    -    |用于得到单位面积谱效率的理论值|
|cdf.py       |-        |用于画概率累计分布函数图|
|dfs_dict_by_distance.py|-|实现深度优先搜索算法|
|dim2_distance.py|-|计算基站和用户之间的欧式距离|
|pc_theory.py|-|用于得到区域覆盖率的理论值|



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

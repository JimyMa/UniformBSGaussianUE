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

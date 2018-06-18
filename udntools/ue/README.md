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

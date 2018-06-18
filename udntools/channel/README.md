### "channel" 子包

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

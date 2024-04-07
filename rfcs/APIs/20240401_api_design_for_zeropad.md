# paddle.nn.ZeroPad设计文档

|API名称 | paddle.nn.ZeroPad1D/paddle.nn.ZeroPad3D | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | zhangyiyuan1112 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2024-04-01 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 提交的markdown设计文档文件名称，如：20240401_api_design_for_zeropad.md<br> | 

# 一、概述
## 1、相关背景
对输入张量边界进行填充是深度学习中常用的操作，常用于卷积神经网络中，以保持特征图的大小。目前飞桨框架中已经支持了`paddle.nn.ZeroPad2D`函数，但是这个函数是一个用于张量2D填充的函数，不支持对1D和3D维度进行填充。因此，本文档提出了`paddle.nn.ZeroPad1D`和`paddle.nn.ZeroPad3D`两个API，用于对张量的最后一个维度和最后三个维度进行零填充。
## 2、功能目标
- 用零填充输入张量边界，1D填充最后一个维度，3D填充最后三个维度。
- 调用形式
  - paddle.nn.ZeroPad1d
  - paddle.nn.ZeroPad3d
## 3、意义
飞浆支持ZeroPad1D和ZeroPad3D组网API。

# 二、飞桨现状
飞桨目前支持ZeroPad2D，但是不支持ZeroPad1D和ZeroPad3D。ZeroPad2D的实现如下：

```python
class ZeroPad2D(Layer):
    """
    This interface is used to construct a callable object of the ``ZeroPad2D`` class.
    Pads the input tensor boundaries with zero.

    Parameters:
        padding (Tensor | List[int] | int):The padding size with data type int.If is int, use thesame padding in all dimensions. Else [len(padding)/2] dimensions of input will be padded.The pad has the form (pad_left, pad_right, pad_top, pad_bottom).
        data_format (str): An string from: "NCHW", "NHWC". Specify the data format of the input data.Default is  "NCHW"
        name (str, optional) : The default value is None.Normally there is no need for user to set this property.For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - x(Tensor): The input tensor of zeropad2d operator, which is a 4-D tensor.The data type can be float32, float64.
        - output(Tensor): The output tensor of zeropad2d operator, which is a 4-D tensor.The data type is same as input x.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_shape = paddle.to_tensor([1, 1, 2, 3])
            >>> pad = [1, 0, 1, 2]
            >>> data = paddle.arange(paddle.prod(input_shape), dtype="float32").reshape(input_shape) + 1
            >>> my_pad = nn.ZeroPad2D(padding=pad)
            >>> result = my_pad(data)
            >>> print(result)
            Tensor(shape=[1, 1, 5, 4],dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0., 0., 0., 0.],
               [0., 1., 2., 3.],
               [0., 4., 5., 6.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]]]])
    """

    def __init__(self, padding, data_format="NCHW", name=None):
        super().__init__()
        self._pad = _npairs(padding, 2)
        self._mode = 'constant'
        self._value = 0.0
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self):
        name_str = f', name={self._name}' if self._name else ''
        return f'padding={self._pad}, data_format={self._data_format}{name_str}'
```

# 三、业内方案调研

## Pytorch方案

PyTorch中有API[`torch.nn.ZeroPad1d()`](https://pytorch.org/docs/2.1/generated/torch.nn.ZeroPad1d.html#zeropad1d)和[`torch.nn.ZeroPad3d()`](https://pytorch.org/docs/2.1/generated/torch.nn.ZeroPad3d.html#zeropad3d)。

ZeroPad1d的实现如下：

```python
class ZeroPad1d(ConstantPad1d):

    padding: Tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        super().__init__(padding, 0.)

    def extra_repr(self) -> str:
        return f'{self.padding}'
```
ZeroPad1D继承了ConstantPad1D，在实例化时，保持padding尺寸不变，并将padding值设置为0。

ConstantPad1D的实现如下：

```python
class ConstantPad1d(_ConstantPadNd):

    padding: Tuple[int, int]

    def __init__(self, padding: _size_2_t, value: float):
        super().__init__(value)
        self.padding = _pair(padding)
```
ConstantPad1D继承了_ConstantPadNd，实现对张量最后一个维度的常量填充。

_ConstantPadNd的实现如下：

```python
class _ConstantPadNd(Module):
    __constants__ = ['padding', 'value']
    value: float
    padding: Sequence[int]

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, 'constant', self.value)

    def extra_repr(self) -> str:
        return f'padding={self.padding}, value={self.value}'
```
_ConstantPadNd的forward函数调用F.pad函数，将padding、value传入，实现对张量的常量填充。

ZeroPad3D的实现方式与ZeroPad1D类似。

综上，pytorch实现ZeroPad的核心方式其实是通过`F.pad`函数实现的，只是在ZeroPad1D和ZeroPad3D中对`F.pad`的维度和填充方式进行了限制。

## TensorFlow方案

tensorflow中有API[`tf.keras.layers.ZeroPadding1D()`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D)和[`tf.keras.layers.ZeroPadding3D()`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D)。

ZeroPadding1D的实现如下：

```python
@keras_export("keras.layers.ZeroPadding1D")
class ZeroPadding1D(Layer):
    """Zero-padding layer for 1D input (e.g. temporal sequence).

    Args:
        padding: Int, or tuple of int (length 2), or dictionary.
            - If int: how many zeros to add at the beginning and end of
              the padding dimension (axis 1).
            - If tuple of 2 ints: how many zeros to add at the beginning and the
              end of the padding dimension (`(left_pad, right_pad)`).

    Input shape:
        3D tensor with shape `(batch_size, axis_to_pad, features)`

    Output shape:
        3D tensor with shape `(batch_size, padded_axis, features)`
    """

    def __init__(self, padding=1, **kwargs):
        super().__init__(**kwargs)
        self.padding = argument_validation.standardize_tuple(
            padding, 2, "padding", allow_zero=True
        )
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if output_shape[1] is not None:
            output_shape[1] += self.padding[0] + self.padding[1]
        return tuple(output_shape)

    def call(self, inputs):
        all_dims_padding = ((0, 0), self.padding, (0, 0))
        return ops.pad(inputs, all_dims_padding)

    def get_config(self):
        config = {"padding": self.padding}
        base_config = super().get_config()
        return {**base_config, **config}
```

ZeroPadding1D继承了Layer，实现了`compute_output_shape`和`call`函数。`compute_output_shape`函数计算输出张量的形状，`call`函数调用`ops.pad`函数实现对张量的填充。只能填充3D tensor的第二个维度（axis=1）。

ZeroPadding3D的实现方式与ZeroPadding1D类似,也是依赖在`call`函数中调用`ops.pad`函数。


# 四、对比分析
pytorch和tensorflow中的ZeroPad1D和ZeroPad3D实现方式类似，都是通过调用`F.pad`或`ops.pad`这种填充函数实现的。但是pytorch对输入数据维度没有严格限制，而tensorflow对输入数据维度有限制。此外tensorflow默认填充中间的维度，而pytorch默认填充最后几个维度。

# 五、设计思路与实现方案

## 命名与参数设计
参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)
## 底层OP设计
## API实现方案

# 六、测试和验收的考量
参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)

# 七、可行性分析和排期规划
时间和开发排期规划，主要milestone

# 八、影响面
需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响

# 名词解释

# 附件及参考资料

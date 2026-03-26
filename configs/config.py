"""
配置文件示例

将常用的配置参数集中管理，方便复用和修改。
"""
import torch


class BaseConfig:
    """基础配置"""

    # data
    root_path = './'
    data_path = 'dataset/weather.csv'
    features = 'M'  # M=多变量, S=单变量, MS=多变量→单变量
    target = 'OT'
    freq = '10min'  # 时间频率: 10min, h, d, t 等

    # seq
    seq_len = 336     # 输入序列长度
    label_len = 336   # 解码器起始长度（通常等于 seq_len）
    pred_len = 96    # 预测长度

    # model
    model = 'Autoformer'  # Autoformer, Transformer, Informer, Reformer
    enc_in = 21      # 编码器输入维度（特征数）
    dec_in = 21      # 解码器输入维度
    c_out = 21       # 输出维度
    d_model = 512    # 模型维度
    n_heads = 8      # 注意力头数
    e_layers = 2     # 编码器层数
    d_layers = 1     # 解码器层数
    d_ff = 2048      # 前馈网络维度
    factor = 1       # 注意力因子
    moving_avg = 25  # 移动平均窗口
    dropout = 0.05   # Dropout 率
    embed = 'timeF'  # 时间嵌入方式: timeF, fixed, learned
    activation = 'gelu'  # 激活函数
    output_attention = False

    # Reformer
    bucket_size = 4
    n_hashes = 4

    # training
    is_training = 1
    train_epochs = 10
    batch_size = 32
    learning_rate = 0.0001
    patience = 10     # Early stopping 耐心值

    # gpu
    use_gpu = True
    gpu = 0
    use_multi_gpu = False
    devices = '0'
    checkpoints = './checkpoints/'

    # time_encoder
    # 0: 手动提取 [month, day, weekday, hour, minute]
    # 1: 使用 time_features 函数
    timeenc = 0


class SmallConfig(BaseConfig):
    """轻量级模型配置（快速训练）"""

    d_model = 128
    n_heads = 4
    e_layers = 2
    d_layers = 1
    d_ff = 256
    batch_size = 64
    train_epochs = 5


class LargeConfig(BaseConfig):
    """大型模型配置（更高精度）"""

    d_model = 1024
    n_heads = 16
    e_layers = 4
    d_layers = 2
    d_ff = 4096
    batch_size = 16
    train_epochs = 20


class WeatherConfig(BaseConfig):
    """天气数据集专用配置"""

    # Weather 数据集有 21 个特征
    enc_in = 21
    dec_in = 21
    c_out = 21

    # 10分钟频率数据
    freq = '10min'
    seq_len = 336    # 56小时 (336 * 10min)
    label_len = 96
    pred_len = 96   # 预测未来16小时

    # 模型参数
    d_model = 512
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = 2048


def get_config(name='base'):
    """获取配置对象"""
    configs = {
        'base': BaseConfig,
        'small': SmallConfig,
        'large': LargeConfig,
        'weather': WeatherConfig,
    }
    return configs.get(name, BaseConfig)()


if __name__ == '__main__':

    # 获取配置
    config = get_config('weather')

    # 打印配置
    print("配置参数")
    print(f"模型: {config.model}")
    print(f"数据: {config.data_path}")
    print(f"序列长度: seq_len={config.seq_len}, pred_len={config.pred_len}")
    print(f"模型维度: d_model={config.d_model}, n_heads={config.n_heads}")
    print(f"批次大小: {config.batch_size}")
    print(f"学习率: {config.learning_rate}")

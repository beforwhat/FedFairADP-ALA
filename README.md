# FedFairADP-ALA 结构
FedFairADP-ALA/
├── configs/                     # 所有实验配置文件（YAML/JSON）
│   ├── datasets/
│   │   ├── cifar10.yaml
│   │   ├── emnist.yaml
│   │   ├── svhn.yaml
│   │   └── femnist.yaml
│   └── methods/
│       ├── fedfairadp_ala.yaml
│       ├── fedavg.yaml
│       ├── dp_fedavg.yaml
│       ├── fedprox.yaml
│       ├── ditto.yaml
│       ├── fedshap.yaml
│       ├── qffl.yaml
│       └── adaptive_clipping.yaml
│
├── data/                        # 数据预处理与划分脚本（不存原始数据）
│   ├── __init__.py
│   ├── dataset_factory.py       # 统一数据集加载接口
│   ├── non_iid_splitter.py      # Dirichlet / 按书写者 / 稀疏标签等Non-IID划分逻辑
│   └── README.md                # 数据下载与预处理说明
│
├── models/                      # 模型定义
│   ├── __init__.py
│   ├── vgg11.py
│   ├── custom_cnn.py            # EMNIST/FEMNIST用
│   └── model_utils.py           # 模型初始化、参数同步工具
│
├── core/                        # 核心算法模块（高内聚、低耦合）
│   ├── __init__.py
│   ├── server.py                # 全局聚合逻辑（含Shapley聚合）
│   ├── client.py                # 客户端本地训练（含ALA + 伪标签）
│   ├── dp_mechanism.py          # 自适应裁剪DP实现（含Shapley驱动阈值调整）
│   ├── fairness_selector.py     # 公平客户端选择（多样性+频率）
│   └── shapley_estimator.py     # 分组蒙特卡洛Shapley值估算（高效版）
│
├── baselines/                   # 基线方法实现（独立模块，避免污染主逻辑）
│   ├── __init__.py
│   ├── fedavg.py
│   ├── dp_fedavg.py
│   ├── fedprox.py
│   ├── ditto.py
│   ├── fedshap.py
│   ├── adaptive_clipping.py
│   └── qffl.py
│
├── utils/                       # 通用工具
│   ├── metrics.py               # 准确率、方差、基尼系数、L2距离等指标计算
│   ├── privacy_checker.py       # DLG攻击模拟、PSNR计算
│   ├── logger.py                # 统一日志与结果记录（CSV/TensorBoard）
│   └── misc.py                  # 随机种子、设备设置等
│
├── experiments/                 # 实验入口脚本（按模块组织）
│   ├── run_performance.py       # 模块1：基础性能对比
│   ├── run_privacy.py           # 模块2：隐私-效用权衡
│   ├── run_ablation.py          # 模块3：消融实验
│   ├── run_fairness.py          # 模块4：公平性验证
│   └── run_efficiency.py        # 模块5：效率与鲁棒性
│
├── results/                     # 自动生成：存放实验输出（.csv, .png, checkpoints）
│
├── requirements.txt             # ✅ 关键！完整依赖列表
├── environment.yml              # （可选）Conda环境文件
├── README.md                    # 项目总览 + 快速启动指南
└── LICENSE                      # MIT/Apache-2.0（推荐MIT）
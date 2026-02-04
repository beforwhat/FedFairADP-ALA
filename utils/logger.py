# utils/logger.py

import os
import csv
import json
from typing import Dict, Any, Optional
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class FederatedLogger:
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True
    ):
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self.config_path = os.path.join(self.log_dir, "config.json")
        self.use_tb = use_tensorboard and SummaryWriter is not None
        if self.use_tb:
            self.tb_writer = SummaryWriter(log_dir=self.log_dir)
        self._write_header()

    def log_config(self, config: Dict[str, Any]):
        """保存实验配置"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """记录标量指标"""
        # CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["step"] + list(metrics.keys()))
            row = {"step": step, **metrics}
            writer.writerow(row)

        # TensorBoard
        if self.use_tb:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)

    def _write_header(self):
        """初始化 CSV 文件头"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write("step\n")  # 后续自动追加字段

    def close(self):
        if self.use_tb:
            self.tb_writer.close()

    def save_final_results(self, results: Dict[str, Any]):
        """保存最终结果（如 ε, 最终准确率等）"""
        path = os.path.join(self.log_dir, "final_results.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
### 任务
1. 实现RTF数据的dataloader；
2. 完成svae和svdd分别在OCL和RTF数据集上的baseline；

#### Notes
在对于OCL和RTF数据集，config中增加`self.img_size = (320, 1440)`，用于说明输入数据的尺寸。

#### Results
**RTF-svae**

log_path: `/data/szy4017/code/IOC/experiments_logs/RTF/svae/_seed_1/logs_18_03_2024_13_28_45.log`

metrics:

| AUC     | AP      |
| ------- | ------- |
| 86.36%   | 97.51% |

**RTF-svdd**

log_path: `/data/szy4017/code/IOC/experiments_logs/RTF/svdd/_seed_1/logs_18_03_2024_13_46_59.log`

metrics:

| AUC     | AP      |
| ------- | ------- |
| 59.09%   | 91.99% |

**OCL-svae**

log_path: `/data/szy4017/code/IOC/experiments_logs/RTF/svae/_seed_1/logs_18_03_2024_13_28_45.log`

metrics:

| AUC     | AP      |
| ------- | ------- |
| 86.36%   | 97.51% |

**OCL-svdd**

log_path: `/data/szy4017/code/IOC/experiments_logs/RTF/svdd/_seed_1/logs_18_03_2024_13_46_59.log`

metrics:

| AUC     | AP      |
| ------- | ------- |
| 59.09%   | 91.99% |
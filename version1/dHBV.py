import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HBVCell(nn.Module):
    def __init__(self, device):
        super(HBVCell, self).__init__()
        self.device = device
        # learnable normalized parameters
        self.params = nn.ParameterDict({
            'BETA': nn.Parameter(torch.tensor(0.5, device=device)),
            'FC':   nn.Parameter(torch.tensor(0.5, device=device)),
            'K0':   nn.Parameter(torch.tensor(0.5, device=device)),
            'K1':   nn.Parameter(torch.tensor(0.5, device=device)),
            'K2':   nn.Parameter(torch.tensor(0.5, device=device)),
            'LP':   nn.Parameter(torch.tensor(0.5, device=device)),
            'PERC': nn.Parameter(torch.tensor(0.5, device=device)),
            'UZL':  nn.Parameter(torch.tensor(0.5, device=device)),
            'TT':   nn.Parameter(torch.tensor(0.5, device=device)),
            'CFMAX':nn.Parameter(torch.tensor(0.5, device=device)),
            'CFR':  nn.Parameter(torch.tensor(0.5, device=device)),
            'CWH':  nn.Parameter(torch.tensor(0.5, device=device)),
            # A and THETA reserved
        })

        # physical ranges for each parameter
        self.ranges = {
            'BETA':   (1.0, 6.0),
            'FC':     (50.0, 1000.0),
            'K0':     (0.05, 0.9),
            'K1':     (0.01, 0.5),
            'K2':     (0.001, 0.2),
            'LP':     (0.2, 1.0),
            'PERC':   (0.0, 10.0),
            'UZL':    (0.0, 100.0),
            'TT':     (-2.5, 2.5),
            'CFMAX':  (0.5, 10.0),
            'CFR':    (0.0, 0.1),
            'CWH':    (0.0, 0.2)
        }

    def scale(self):
        # map normalized [0,1] to [min,max]
        scaled = {}
        for k,p in self.params.items():
            lo, hi = self.ranges[k]
            v = torch.clamp(p, 0.0, 1.0) * (hi - lo) + lo
            scaled[k] = v
        return scaled

    def forward(self, step_in, states):
        T = step_in[:, 0:1]
        P = step_in[:, 1:2]
        PET = step_in[:, 2:3]

        snow, meltw, sm, suz, slz = states.split(1, dim=1)
        # get physical params
        sp = self.scale()
        # 1. rain/snow split
        rain = torch.where(T >= sp['TT'], P, torch.zeros_like(P))
        snow_in = P - rain
        snow = snow + snow_in
        # 2. snowmelt
        melt_pot = sp['CFMAX'] * torch.clamp(T - sp['TT'], min=0.0)
        melt_act = torch.min(melt_pot, snow)
        snow = snow - melt_act
        meltw = meltw + melt_act
        # 3. refreezing
        refreeze = sp['CFR'] * sp['CFMAX'] * torch.clamp(sp['TT'] - T, min=0.0)
        refreeze = torch.min(refreeze, meltw)
        snow = snow + refreeze
        meltw = meltw - refreeze
        # 4. to soil
        to_soil = meltw - sp['CWH'] * snow
        to_soil = torch.clamp(to_soil, min=0.0)
        meltw = meltw - to_soil
        # 5. soil moisture update
        wetness = torch.clamp((sm / sp['FC']) ** sp['BETA'], 0.0, 1.0)
        recharge = (rain + to_soil) * wetness
        sm = sm + rain + to_soil - recharge
        excess = torch.clamp(sm - sp['FC'], min=0.0)
        sm = sm - excess
        # 6. actual evapotranspiration
        evap_factor = torch.clamp(sm / (sp['LP'] * sp['FC']), 0.0, 1.0)
        et_act = torch.min(sm, PET * evap_factor)
        sm = sm - et_act
        # 7. groundwater reservoirs
        suz = suz + recharge + excess
        perc = torch.min(suz, sp['PERC'])
        suz = suz - perc
        q0 = sp['K0'] * torch.clamp(suz - sp['UZL'], min=0.0)
        suz = suz - q0
        q1 = sp['K1'] * suz
        suz = suz - q1
        slz = slz + perc
        q2 = sp['K2'] * slz
        slz = slz - q2
        # outflow
        U = q0 + q1 + q2
        next_states = torch.cat([snow, meltw, sm, suz, slz], dim=1)
        return U, next_states

class HBVLayer(nn.Module):
    def __init__(self, device=None):
        super(HBVLayer, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hbv_cell = HBVCell(device=self.device)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        states = torch.full((batch_size, 5), 0.001, device=self.device)

        outputs = []
        for t in range(seq_len):
            step_in = inputs[:, t, :]
            out, states = self.hbv_cell(step_in, states)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)
        return outputs

def generate_train_test(train_set, test_set, wrap_length):
    train_x_np = train_set.values[:, :-1]
    train_y_np = train_set.values[:, -1:]
    test_x_np = test_set.values[:, :-1]
    test_y_np = test_set.values[:, -1:]

    wrap_number_train = (train_set.shape[0] - wrap_length) // 365 + 1

    train_x = np.empty(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_y = np.empty(shape=(wrap_number_train, wrap_length, train_y_np.shape[1]))

    test_x = np.expand_dims(test_x_np, axis=0)
    test_y = np.expand_dims(test_y_np, axis=0)

    for i in range(wrap_number_train):
        train_x[i, :, :] = train_x_np[i * 365:(wrap_length + i * 365), :]
        train_y[i, :, :] = train_y_np[i * 365:(wrap_length + i * 365), :]

    return train_x, train_y, test_x, test_y

def nse(predictions, targets):
    return 1 - (torch.sum((targets - predictions) ** 2) / torch.sum((targets - torch.mean(targets)) ** 2))


def nse_loss(predictions, targets):
    nse_val = nse(predictions, targets)
    return 1 - nse_val

def process_site(file_path, logs_dir, plots_dir, validation_data_dir, results_file, wrap_length=2190, num_epochs=100):
    try:
        # 获取站点名称（从文件名中提取）
        filename = os.path.splitext(os.path.basename(file_path))[0]
        parts = filename.split('_')
        if len(parts) >= 5:
            site_name = parts[4]
        else:
            site_name = 'unknown_site'

        # 设置日志记录
        log_file = os.path.join(logs_dir, f"{site_name}.log")
        logger = logging.getLogger(site_name)
        logger.setLevel(logging.INFO)
        # 避免重复添加处理器
        if not logger.handlers:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        logger.info(f"开始处理站点: {site_name}")

        # 读取数据
        data = pd.read_csv(file_path)
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
        data = data.dropna()

        # # 按照指定的日期划分数据集，添加暖启动期
        train_data = data[(data['date'] >= '1975-01-01') & (data['date'] <= '1997-12-31')]
        val_warmup_data = data[(data['date'] >= '1998-01-01') & (data['date'] <= '1998-12-31')]
        val_data = data[(data['date'] >= '1999-01-01') & (data['date'] <= '2014-12-31')]
        val_data_with_warmup = pd.concat([val_warmup_data, val_data], ignore_index=True)

        # 提取训练数据
        Temp_train = train_data['temperature'].values
        Prec_train = train_data['precipitation'].values
        Evap_train = train_data['pet'].values
        Qobs_train = train_data['discharge_spec'].values
        dates_train = train_data['date'].values

        # 提取验证数据（包括暖启动期）
        Temp_val = val_data_with_warmup['temperature'].values
        Prec_val = val_data_with_warmup['precipitation'].values
        Evap_val = val_data_with_warmup['pet'].values
        Qobs_val = val_data_with_warmup['discharge_spec'].values
        dates_val = val_data_with_warmup['date'].values

        # 创建训练和验证集
        train_set = pd.DataFrame({'Temp': Temp_train, 'Prec': Prec_train, 'Evap': Evap_train, 'Qobs': Qobs_train})
        test_set = pd.DataFrame({'Temp': Temp_val, 'Prec': Prec_val, 'Evap': Evap_val, 'Qobs': Qobs_val})

        # 调用数据生成函数
        train_x, train_y, test_x, test_y = generate_train_test(train_set, test_set, wrap_length)

        # 转换为Tensor
        train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
        train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
        test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
        test_y = torch.tensor(test_y, dtype=torch.float32).to(device)

        model = HBVLayer(device=device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=5, min_lr=0.0001,
                                                         verbose=True)

        early_stopping_patience = 20
        early_stopping_counter = 0
        best_val_nse = -float('inf')

        # 记录训练开始时间
        train_start_time = time.time()
        cumulative_time = 0  # 累计时间

        # 模型训练
        best_model_wts = None

        for epoch in range(num_epochs):
            epoch_start_time = time.time()  # 记录每个epoch的开始时间

            model.train()
            optimizer.zero_grad()
            outputs = model(train_x)
            loss = nse_loss(outputs, train_y)
            loss.backward()
            optimizer.step()

            # 获取训练损失
            with torch.no_grad():
                outputs = model(train_x)
                train_loss = nse_loss(outputs, train_y).item()

            # 验证集评估
            model.eval()
            with torch.no_grad():
                val_outputs_full = model(test_x)  # 包含暖启动期的预测
                # 去除暖启动期的数据，用于计算损失和指标
                warmup_length = len(val_warmup_data)
                val_outputs = val_outputs_full[:, warmup_length:, :]
                val_targets = test_y[:, warmup_length:, :]
                val_loss = nse_loss(val_outputs, val_targets)
                val_nse = nse(val_outputs, val_targets).item()

            scheduler.step(val_nse)

            # 保存最佳模型
            if val_nse > best_val_nse:
                best_val_nse = val_nse
                best_model_wts = model.state_dict()
                logger.info(f"最佳模型权重已保存在第 {epoch + 1} 轮")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                logger.info("早停机制触发")
                break

            # 计算R2分数
            train_nse = nse(outputs, train_y).item()
            train_r2 = r2_score(train_y.squeeze().detach().cpu().numpy(),
                                outputs.squeeze().detach().cpu().numpy())
            val_r2 = r2_score(val_targets.squeeze().detach().cpu().numpy(),
                              val_outputs.squeeze().detach().cpu().numpy())

            # 记录时间
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            cumulative_time += epoch_time

            logger.info(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss.item():.4f}, '
                f'Train NSE: {train_nse:.4f}, Val NSE: {val_nse:.4f}, Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}, '
                f'Epoch Time: {epoch_time:.2f}s, Cumulative Time: {cumulative_time:.2f}s')

        # 记录训练结束时间
        train_end_time = time.time()

        # 计算并记录训练时间
        train_time = train_end_time - train_start_time
        logger.info(f"总训练时间: {train_time:.2f} 秒")

        # 载入最佳模型权重
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            logger.info("已载入最佳模型权重")
        else:
            logger.info("未找到最佳模型权重")


        # 验证结果可视化
        model.eval()
        with torch.no_grad():
            predicted_full = model(test_x).squeeze().detach().cpu().numpy()
            actual_full = test_y.squeeze().detach().cpu().numpy()

            # 去除暖启动期的数据
            predicted = predicted_full[warmup_length:]
            actual = actual_full[warmup_length:]
            dates_val_actual = dates_val[warmup_length:]

        plt.figure(figsize=(10, 6))
        plt.plot(dates_val_actual, actual, label='Observed')
        plt.plot(dates_val_actual, predicted, label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Discharge')
        plt.title(f'Observed vs Predicted Discharge for Site {site_name}')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"{site_name}.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"对比图已保存: {plot_path}")

        # 保存验证数据
        validation_df = pd.DataFrame({
            'date': dates_val_actual,
            'observed': actual,
            'predicted': predicted
        })
        validation_site_dir = os.path.join(validation_data_dir, site_name)
        os.makedirs(validation_site_dir, exist_ok=True)
        validation_df.to_csv(os.path.join(validation_site_dir, f"{site_name}_validation.csv"), index=False)
        logger.info(f"验证数据已保存到: {validation_site_dir}")

        # 保存校准结果
        with open(results_file, 'a') as f:
            f.write(f"{site_name},{best_val_nse:.4f},{train_time:.2f}\n")

        logger.info(f"站点 {site_name} 处理完成\n")

    except Exception as e:
        logger.error(f"处理站点 {file_path} 时发生错误: {e}")


def main():
    # 设置数据文件夹路径
    data_folder = '/HBV可微版本'

    # 设置输出文件夹路径
    output_folder = os.path.join(data_folder, 'D_HBV')
    logs_dir = os.path.join(output_folder, 'logs')
    plots_dir = os.path.join(output_folder, 'plots')
    validation_data_dir = os.path.join(output_folder, 'validation_data')  # 新增的文件夹路径
    results_file = os.path.join(output_folder, 'calibration_results.csv')  # 新增的结果文件路径

    # 创建输出文件夹及子文件夹
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(validation_data_dir, exist_ok=True)

    # 初始化结果文件
    with open(results_file, 'w') as f:
        f.write("Site,NSE,TrainingTime(s)\n")

    # 遍历数据文件夹中的所有CSV文件
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_folder, file_name)
            process_site(file_path, logs_dir, plots_dir, validation_data_dir, results_file)

    print("所有站点文件已处理完成。")


if __name__ == "__main__":
    main()
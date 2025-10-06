import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import json
import os
import pickle

# 自定义数据集类（添加缩放功能）
class MultiTaskDataset(Dataset):
    def __init__(self, csv_path, selected_tasks=None, scaler=None):
        self.data = pd.read_csv(csv_path)
        
        # 如果指定了要选择的任务，进行过滤
        if selected_tasks is not None:
            self.data = self.data[self.data['task'].isin(selected_tasks)]
            print(f"筛选任务: {selected_tasks}")
        
        # 自动识别特征列（排除index, label, task列）
        non_feature_cols = ['index', 'label', 'task']
        all_cols = self.data.columns.tolist()
        self.feature_cols = [col for col in all_cols if col not in non_feature_cols]
        
        # 获取特征数据
        features = self.data[self.feature_cols].values.astype(np.float32)
        
        # 特征缩放
        if scaler is not None:
            self.scaler = scaler
            self.features = scaler.transform(features).astype(np.float32)
            print("✅ 应用预训练缩放器进行特征缩放")
        else:
            self.scaler = None
            self.features = features
            print("⚠️ 未进行特征缩放")
        
        self.labels = self.data['label'].values.astype(np.float32)
        self.tasks = self.data['task'].values
        
        print(f"加载测试数据: {csv_path}")
        print(f"特征列数: {len(self.feature_cols)}")
        print(f"样本数量: {len(self.features)}")
        print(f"任务分布: {pd.Series(self.tasks).value_counts().to_dict()}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.tasks[idx]

# 多任务分类模型（与训练代码保持一致）
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, task_names):
        super(MultiTaskModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            prev_dim = hidden_dim
        self.shared_backbone = nn.Sequential(*layers)

        # 每个任务一个输出头
        self.task_heads = nn.ModuleDict({task: nn.Linear(prev_dim, 1) for task in task_names})

    def forward(self, x, task_id):
        shared_output = self.shared_backbone(x)
        logit = self.task_heads[task_id](shared_output)
        return logit.squeeze(1)

def load_model(model_path, device):
    """加载训练好的模型"""
    # 使用 weights_only=False 来加载模型
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 手动设置正确的隐藏层维度
    config = checkpoint['config']
    config['hidden_dims'] = [256, 128] # 修改为正确的隐藏层维度
    
    print(f"使用隐藏层配置: {config['hidden_dims']}")
    
    # 使用修改后的配置创建模型
    model = MultiTaskModel(
        input_dim=config['feature_dim'],
        hidden_dims=config['hidden_dims'],
        task_names=config['task_names']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def load_scaler(scaler_path):
    """加载缩放器"""
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ 缩放器已从 {scaler_path} 加载")
        return scaler
    except FileNotFoundError:
        print(f"⚠️ 未找到缩放器文件: {scaler_path}")
        return None
    except Exception as e:
        print(f"❌ 加载缩放器时出错: {e}")
        return None

def predict(model, loader, device, task_names):
    """进行预测并返回结果"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for features, labels, tasks in loader:
            features = features.to(device)
            labels = labels.numpy()
            tasks = np.array(tasks)
            
            # 按任务批量处理，提高效率
            for task in task_names:
                # 创建任务掩码
                task_mask = (tasks == task)
                
                if np.any(task_mask):
                    task_features = features[task_mask]
                    task_labels = labels[task_mask]
                    task_task_names = tasks[task_mask]
                    
                    logits = model(task_features, task)
                    pred_probs = torch.sigmoid(logits).cpu().numpy()
                    pred_labels = (pred_probs >= 0.5).astype(int)
                    
                    # 收集预测结果
                    for i in range(len(task_labels)):
                        all_predictions.append({
                            'true_label': int(task_labels[i]),
                            'pred_label': int(pred_labels[i]),
                            'pred_prob': float(pred_probs[i]),
                            'task': task_task_names[i]
                        })
    
    return all_predictions

def evaluate_predictions(predictions, task_names):
    """评估预测结果"""
    results = {}
    
    for task in task_names:
        # 筛选该任务的预测结果
        task_preds = [p for p in predictions if p['task'] == task]
        
        if not task_preds:
            results[task] = {
                'AUC': 0.5,
                'F1': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'Accuracy': 0.0,
                'Support': 0
            }
            continue
        
        true_labels = np.array([p['true_label'] for p in task_preds])
        pred_probs = np.array([p['pred_prob'] for p in task_preds])
        pred_labels = np.array([p['pred_label'] for p in task_preds])
        
        # 计算各项指标
        accuracy = np.mean(true_labels == pred_labels)
        
        if len(np.unique(true_labels)) > 1:
            auc = roc_auc_score(true_labels, pred_probs)
            f1 = f1_score(true_labels, pred_labels, zero_division=0)
            precision = precision_score(true_labels, pred_labels, zero_division=0)
            recall = recall_score(true_labels, pred_labels, zero_division=0)
        else:
            auc, f1, precision, recall = 0.5, 0.0, 0.0, 0.0
        
        results[task] = {
            'AUC': auc,
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy,
            'Support': len(task_preds)
        }
    
    return results

def save_test_results(predictions, results, save_path):
    """保存测试结果"""
    # 保存详细预测结果
    detailed_df = pd.DataFrame(predictions)
    detailed_file = os.path.join(save_path, 'detailed_predictions.csv')
    detailed_df.to_csv(detailed_file, index=False)
    
    # 保存汇总指标
    summary_df = pd.DataFrame.from_dict(results, orient='index')
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={'index': 'task'}, inplace=True)
    summary_file = os.path.join(save_path, 'test_results_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    # 保存JSON格式的结果
    json_file = os.path.join(save_path, 'test_results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"测试结果已保存到: {save_path}")

def main():
    default_input = f"./data/test_features.csv"
    parser = argparse.ArgumentParser(description="Run TXSelect evaluation with optional custom paths")
    parser.add_argument("--input", dest="input_path", default=default_input,
                        help="Path to the test CSV; defaults to the project test data")
    parser.add_argument("--output", dest="output_dir", default="./results/",
                        help="Directory to store evaluation outputs; defaults to ./results/")
    args = parser.parse_args()

    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 模型路径和测试数据路径
    model_path = "./models/TXSelect.pth"
    scaler_path = "./models/scaler.pkl"
    test_csv_path = args.input_path
    output_dir = args.output_dir

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载模型
        model, checkpoint = load_model(model_path, device)
        config = checkpoint['config']
        
        print("模型配置信息:")
        print(f"特征维度: {config['feature_dim']}")
        print(f"任务列表: {config['task_names']}")
        print(f"隐藏层: {config['hidden_dims']}")
        print(f"训练时的最佳F1: {checkpoint.get('best_f1', 'N/A'):.4f}")
        
        # 检查配置中是否有缩放信息
        if 'feature_scaling' in config:
            print(f"特征缩放方式: {config['feature_scaling']}")
        
        # 加载缩放器
        scaler = load_scaler(scaler_path)
        
        # 加载测试数据（应用缩放器）
        test_dataset = MultiTaskDataset(
            test_csv_path, 
            selected_tasks=config['task_names'],
            scaler=scaler
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 进行预测
        print("\n开始预测...")
        predictions = predict(model, test_loader, device, config['task_names'])
        
        # 评估结果
        results = evaluate_predictions(predictions, config['task_names'])
        
        # 打印结果
        print("\n" + "="*60)
        print("测试结果汇总:")
        
        avg_auc = np.mean([results[task]['AUC'] for task in config['task_names']])
        avg_f1 = np.mean([results[task]['F1'] for task in config['task_names']])
        
        print(f"平均AUC: {avg_auc:.4f}")
        print(f"平均F1: {avg_f1:.4f}")
        
        print("\n各任务详细指标:")
        for task in config['task_names']:
            metrics = results[task]
            print(f"\n{task}:")
            print(f"  AUC: {metrics['AUC']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")
            print(f"  F1: {metrics['F1']:.4f}, Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}")
            print(f"  Support: {metrics['Support']}")
        
        # 保存结果
        save_test_results(predictions, results, output_dir)
        
        # 添加缩放信息到结果中
        scaling_info = {
            'feature_scaling_applied': scaler is not None,
            'scaler_path': scaler_path if scaler is not None else 'None',
            'scaling_type': config.get('feature_scaling', 'Unknown') if scaler is not None else 'None'
        }
        
        # 保存包含缩放信息的完整结果
        full_results = {
            'scaling_info': scaling_info,
            'metrics': results,
            'summary': {
                'avg_auc': avg_auc,
                'avg_f1': avg_f1,
                'total_samples': len(predictions)
            }
        }
        
        with open(os.path.join(output_dir, 'full_test_results.json'), 'w') as f:
            json.dump(full_results, f, indent=4)
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

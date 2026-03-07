#!/usr/bin/env python3
"""
分析2019文件夹中数据文件的脚本
分析 his.npz, idx_train.npy, idx_val.npy, idx_test.npy 文件的内容和结构
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

def analyze_npz_file(file_path):
    """分析 .npz 文件"""
    print(f"\n{'='*60}")
    print(f"分析文件: {file_path}")
    print(f"{'='*60}")
    
    # 加载 npz 文件
    data = np.load(file_path)
    
    print(f"文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    print(f"包含的数组键: {list(data.keys())}")
    
    for key in data.keys():
        arr = data[key]
        print(f"\n数组 '{key}':")
        print(f"  - 形状: {arr.shape}")
        print(f"  - 数据类型: {arr.dtype}")
        print(f"  - 内存大小: {arr.nbytes / (1024*1024):.2f} MB")
        
        if arr.ndim > 0:
            print(f"  - 最小值: {np.nanmin(arr):.6f}")
            print(f"  - 最大值: {np.nanmax(arr):.6f}")
            print(f"  - 平均值: {np.nanmean(arr):.6f}")
            print(f"  - 标准差: {np.nanstd(arr):.6f}")
            
            # 检查 NaN 值
            nan_count = np.isnan(arr).sum()
            if nan_count > 0:
                print(f"  - NaN 值数量: {nan_count} ({nan_count/arr.size*100:.2f}%)")
            else:
                print(f"  - 无 NaN 值")
                
            # 显示前几个值作为样例
            if arr.size <= 10:
                print(f"  - 所有值: {arr.flatten()}")
            else:
                print(f"  - 前10个值: {arr.flatten()[:10]}")
                
        # 如果是3维数据，可能是时间序列数据
        if key == 'data' and arr.ndim == 3:
            print(f"  - 可能的含义: (时间步, 节点数, 特征数)")
            print(f"    * 时间步数: {arr.shape[0]}")
            print(f"    * 节点数: {arr.shape[1]}")
            print(f"    * 特征数: {arr.shape[2]}")

def analyze_npy_file(file_path):
    """分析 .npy 文件"""
    print(f"\n{'='*60}")
    print(f"分析文件: {file_path}")
    print(f"{'='*60}")
    
    # 加载 npy 文件
    data = np.load(file_path)
    
    print(f"文件大小: {os.path.getsize(file_path) / 1024:.2f} KB")
    print(f"形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"元素数量: {data.size}")
    
    if data.size > 0:
        print(f"最小值: {data.min()}")
        print(f"最大值: {data.max()}")
        
        # 显示索引值
        print(f"前10个索引: {data[:10]}")
        if data.size > 10:
            print(f"后10个索引: {data[-10:]}")
        
        # 分析索引的连续性
        if data.size > 1:
            diff = np.diff(data)
            print(f"索引差值统计:")
            print(f"  - 最小差值: {diff.min()}")
            print(f"  - 最大差值: {diff.max()}")
            print(f"  - 平均差值: {diff.mean():.2f}")
            
            # 检查是否连续
            is_continuous = np.all(diff == 1)
            print(f"  - 是否连续: {is_continuous}")
            
            if not is_continuous:
                unique_diffs = np.unique(diff)
                print(f"  - 所有差值: {unique_diffs}")

def analyze_dataset_split(train_idx, val_idx, test_idx):
    """分析数据集分割情况"""
    print(f"\n{'='*60}")
    print("数据集分割分析")
    print(f"{'='*60}")
    
    total_samples = len(train_idx) + len(val_idx) + len(test_idx)
    
    print(f"训练集样本数: {len(train_idx)} ({len(train_idx)/total_samples*100:.1f}%)")
    print(f"验证集样本数: {len(val_idx)} ({len(val_idx)/total_samples*100:.1f}%)")
    print(f"测试集样本数: {len(test_idx)} ({len(test_idx)/total_samples*100:.1f}%)")
    print(f"总样本数: {total_samples}")
    
    print(f"\n索引范围:")
    print(f"训练集: {train_idx[0]} - {train_idx[-1]}")
    print(f"验证集: {val_idx[0]} - {val_idx[-1]}")
    print(f"测试集: {test_idx[0]} - {test_idx[-1]}")
    
    # 检查是否有重叠
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    overlap_train_val = len(train_set & val_set)
    overlap_train_test = len(train_set & test_set)
    overlap_val_test = len(val_set & test_set)
    
    print(f"\n重叠检查:")
    print(f"训练集与验证集重叠: {overlap_train_val}")
    print(f"训练集与测试集重叠: {overlap_train_test}")
    print(f"验证集与测试集重叠: {overlap_val_test}")
    
    if overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0:
        print("✅ 数据集分割正确，无重叠")
    else:
        print("❌ 数据集分割有重叠问题")

def main():
    # 定义文件路径
    data_dir = "/home/defu/workspace/LargeST/data/sd/2019"
    
    files = {
        'his.npz': 'npz',
        'idx_train.npy': 'npy',
        'idx_val.npy': 'npy',
        'idx_test.npy': 'npy'
    }
    
    print("SD 2019 数据集文件分析")
    print("=" * 60)
    print(f"分析目录: {data_dir}")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查文件是否存在
    existing_files = []
    for filename in files.keys():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            existing_files.append(filename)
            print(f"✅ 找到文件: {filename}")
        else:
            print(f"❌ 文件不存在: {filename}")
    
    if not existing_files:
        print("没有找到任何文件，请检查路径是否正确。")
        return
    
    # 分析各个文件
    idx_data = {}
    
    for filename in existing_files:
        filepath = os.path.join(data_dir, filename)
        file_type = files[filename]
        
        try:
            if file_type == 'npz':
                analyze_npz_file(filepath)
            elif file_type == 'npy':
                analyze_npy_file(filepath)
                # 保存索引数据用于后续分析
                if 'idx' in filename:
                    idx_data[filename.replace('.npy', '')] = np.load(filepath)
        except Exception as e:
            print(f"❌ 分析文件 {filename} 时出错: {str(e)}")
    
    # 分析数据集分割
    if len(idx_data) == 3:  # 如果所有索引文件都存在
        try:
            analyze_dataset_split(
                idx_data['idx_train'],
                idx_data['idx_val'], 
                idx_data['idx_test']
            )
        except Exception as e:
            print(f"❌ 分析数据集分割时出错: {str(e)}")
    
    print(f"\n{'='*60}")
    print("分析完成！")
    print(f"{'='*60}")
    
    # 文件说明
    print(f"\n文件说明:")
    print(f"- his.npz: 历史时间序列数据，包含 data(时间序列特征), mean(标准化均值), std(标准化标准差)")
    print(f"- idx_train.npy: 训练集的时间索引")
    print(f"- idx_val.npy: 验证集的时间索引")  
    print(f"- idx_test.npy: 测试集的时间索引")

if __name__ == "__main__":
    main()
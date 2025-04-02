#!/bin/bash

# 用法说明
usage() {
    echo "用法: $0 [-d 数据集] [-t 数据类型] [-m 模型] [-a 算法] [-r 轮数] [-i 设备ID]"
    echo "示例:"
    echo "  $0 -d MNIST -t noniid -m CNN -a FedAvg -r 100 -i 0"
    echo "  $0 -d CIFAR10 -t iid -m ResNet -a FedProx -r 50 -i 1"
    exit 1
}

# 默认参数
DATASET="MNIST"
DATA_TYPE="noniid"
MODEL="CNN"
ALGO="FedAvg"
ROUNDS=100
DEVICE_ID=0

# 解析命令行参数
while getopts ":d:t:m:a:r:i:h" opt; do
    case $opt in
        d) DATASET="$OPTARG" ;;
        t) DATA_TYPE="$OPTARG" ;;
        m) MODEL="$OPTARG" ;;
        a) ALGO="$OPTARG" ;;
        r) ROUNDS="$OPTARG" ;;
        i) DEVICE_ID="$OPTARG" ;;
        h) usage ;;
        \?) echo "无效选项: -$OPTARG" >&2; usage ;;
        :) echo "选项 -$OPTARG 需要参数" >&2; usage ;;
    esac
done

# 打印配置信息
echo "=== 运行配置 ==="
echo "数据集: $DATASET"
echo "数据类型: $DATA_TYPE"
echo "模型: $MODEL"
echo "算法: $ALGO"
echo "训练轮数: $ROUNDS"
echo "设备ID: $DEVICE_ID"
echo "================"

# # 数据生成（如果数据集需要）
# if [[ "$DATASET" == "MNIST" ]]; then
#     echo "生成 $DATA_TYPE MNIST 数据集..."
#     python dataset/generate_MNIST.py noniid - dir || {
#         echo "MNIST 数据生成失败!"; exit 1
#     }
# elif [[ "$DATASET" == "CIFAR10" ]]; then
#     echo "生成 $DATA_TYPE CIFAR10 数据集..."
#     python generate_Cifar10.py noniid - dir || {
#         echo "CIFAR10 数据生成失败!"; exit 1
#     }
# # 可以添加更多数据集
# else
#     echo "使用预处理的 $DATASET 数据集"
# fi

# 运行训练
echo "开始训练..."
python ./system/main.py \
    -data $DATASET \
    -m $MODEL \
    -algo $ALGO \
    -gr $ROUNDS \
    -did $DEVICE_ID || {
    echo "训练失败!"; exit 1
}

echo "训练完成!"
python main.py -data Cifar10 -m CNN -algo FedAvg -gr 100 -did 0 -nc 100
python main.py -data Cifar10 -m LeNetCifar -algo FedAvg -gr 200 -did 0 -nc 50
ResNet18
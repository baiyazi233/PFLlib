from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
import subprocess
import threading
import re
import logging
import eventlet
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import uuid
import warnings
import sys
import types
import json

# 过滤PyTorch警告
warnings.filterwarnings("ignore", message="`torch.distributed.reduce_op` is deprecated")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 创建上传目录
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = os.path.join(UPLOAD_FOLDER, 'models')
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# Global variable to store the current training process
current_process = None

# 定义与训练模型兼容的LeNet模型
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class LeNet(nn.Module):
    def __init__(self, feature_dim=50*4*4, bottleneck_dim=256, num_classes=10, iswn=None):
        super(LeNet, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 为了兼容性，定义基础拆分模型类
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()
        self.base = base
        self.head = head
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out

# 创建模拟的flcore模块以便能够加载依赖flcore的模型
logger.info("创建模拟flcore模块...")
if 'flcore' not in sys.modules:
    # 创建一个模拟的模块结构
    
    # 创建主模块
    flcore = types.ModuleType('flcore')
    sys.modules['flcore'] = flcore
    
    # 创建子模块
    trainmodel = types.ModuleType('flcore.trainmodel')
    sys.modules['flcore.trainmodel'] = trainmodel
    flcore.trainmodel = trainmodel
    
    # 创建models模块
    models = types.ModuleType('flcore.trainmodel.models')
    sys.modules['flcore.trainmodel.models'] = models
    trainmodel.models = models
    
    # 添加类到models模块
    models.BaseHeadSplit = BaseHeadSplit
    models.LeNet = LeNet
    models.init_weights = init_weights
    
    logger.info("模拟flcore模块创建成功")

@app.route('/')
def index():
    logger.info("Serving index page")
    return render_template('index.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    global current_process
    
    # Kill any existing training process
    if current_process:
        logger.info("Terminating existing training process")
        current_process.terminate()
        current_process = None
    
    data = request.json
    command = data['command']
    logger.info(f"Starting training with command: {command}")
    
    def run_training():
        global current_process
        logger.info("Running command: %s", command)
        current_process = subprocess.Popen(
            command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 正则表达式适配 main.py 输出
        train_loss_pattern = r"Averaged Train Loss: ([\d\.]+)"
        test_acc_pattern = r"Averaged Test Accurancy: ([\d\.]+)"
        std_acc_pattern = r"Std Test Accurancy: ([\d\.]+)"
        time_pattern = r"time cost ------------------------- ([\d\.]+)"
        
        round_num = 0
        metrics = {'round': 0, 'time': 0, 'loss': 0, 'avg_accuracy': 0, 'std_accuracy': 0}
        
        while True:
            # 检查进程是否被终止
            if current_process is None:
                logger.info("训练进程已被终止")
                break
            
            try:
                line = current_process.stdout.readline()
                if not line and current_process.poll() is not None:
                    logger.info("Training process completed")
                    break
                
                logger.debug("Received line from main.py: %s", line.strip())
                
                train_loss_match = re.search(train_loss_pattern, line)
                test_acc_match = re.search(test_acc_pattern, line)
                std_acc_match = re.search(std_acc_pattern, line)
                time_match = re.search(time_pattern, line)
                
                # 打印匹配结果
                if train_loss_match or test_acc_match or std_acc_match or time_match:
                    logger.debug("Pattern matches found:")
                    logger.debug("Train loss match: %s", train_loss_match.group(1) if train_loss_match else None)
                    logger.debug("Test acc match: %s", test_acc_match.group(1) if test_acc_match else None)
                    logger.debug("Std acc match: %s", std_acc_match.group(1) if std_acc_match else None)
                    logger.debug("Time match: %s", time_match.group(1) if time_match else None)
                
                # 累积每轮的指标，等四个都收集到就推送
                if train_loss_match:
                    metrics['loss'] = float(train_loss_match.group(1))
                    logger.debug("Found train loss: %f", metrics['loss'])
                if test_acc_match:
                    metrics['avg_accuracy'] = float(test_acc_match.group(1))
                    logger.debug("Found test accuracy: %f", metrics['avg_accuracy'])
                if std_acc_match:
                    metrics['std_accuracy'] = float(std_acc_match.group(1))
                    logger.debug("Found std accuracy: %f", metrics['std_accuracy'])
                if time_match:
                    metrics['time'] = float(time_match.group(1))
                    metrics['round'] = round_num
                    logger.info("Emitting metrics to front: %s", metrics)
                    socketio.emit('training_update', metrics.copy())
                    round_num += 1
            except Exception as e:
                if current_process is None:
                    logger.info("进程已终止，退出监控线程")
                    break
                logger.error(f"处理训练输出时发生错误: {str(e)}")
                # 继续循环而不是中断
        
        # 最后检查进程是否还在运行
        if current_process is not None and current_process.poll() is None:
            logger.info("Terminating training process")
            try:
                current_process.terminate()
            except Exception as e:
                logger.error(f"终止进程时发生错误: {str(e)}")
                # 忽略错误继续
    
    # Start training in a separate thread
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success'})

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global current_process
    logger.info("收到停止训练请求")
    
    if current_process:
        logger.info("停止训练进程")
        try:
            # 先保存一个引用以便终止进程
            process_to_terminate = current_process
            # 先将global变量置为None，这样监控线程就会安全退出
            current_process = None
            # 然后终止进程
            process_to_terminate.terminate()
            logger.info("训练进程已终止")
        except Exception as e:
            logger.error(f"终止进程时发生错误: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    else:
        logger.info("没有正在运行的训练进程")
    
    return jsonify({'status': 'success', 'message': '训练已停止'})

# 上传模型文件
@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No model file uploaded'}), 400
    
    model_file = request.files['model_file']
    if model_file.filename == '':
        return jsonify({'status': 'error', 'message': 'No model file selected'}), 400
    
    if not model_file.filename.endswith('.pt'):
        return jsonify({'status': 'error', 'message': 'File must be a .pt file'}), 400
    
    # 生成唯一ID作为文件名，但保留原始文件名
    original_filename = model_file.filename
    filename = str(uuid.uuid4()) + '.pt'
    file_path = os.path.join(app.config['MODEL_FOLDER'], filename)
    model_file.save(file_path)
    
    # 保存原始文件名到映射文件
    name_mapping_file = os.path.join(app.config['MODEL_FOLDER'], 'name_mapping.json')
    name_mapping = {}
    
    if os.path.exists(name_mapping_file):
        try:
            with open(name_mapping_file, 'r') as f:
                name_mapping = json.load(f)
        except:
            name_mapping = {}
    
    name_mapping[filename] = original_filename
    
    with open(name_mapping_file, 'w') as f:
        json.dump(name_mapping, f)
    
    # 返回模型ID和名称
    return jsonify({
        'status': 'success',
        'model_id': filename,
        'original_name': original_filename
    })

# 获取上传的模型列表
@app.route('/get_models', methods=['GET'])
def get_models():
    model_files = []
    
    # 加载文件名映射
    name_mapping_file = os.path.join(app.config['MODEL_FOLDER'], 'name_mapping.json')
    name_mapping = {}
    
    if os.path.exists(name_mapping_file):
        try:
            with open(name_mapping_file, 'r') as f:
                name_mapping = json.load(f)
        except:
            name_mapping = {}
    
    for model_id in os.listdir(app.config['MODEL_FOLDER']):
        if model_id.endswith('.pt'):
            # 获取原始文件名，如果没有则使用ID
            original_name = name_mapping.get(model_id, model_id)
            model_files.append({
                'model_id': model_id,
                'name': original_name
            })
    
    return jsonify({'models': model_files})

# 上传MNIST图片进行推理
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image_file' not in request.files or 'model_id' not in request.form:
        return jsonify({'status': 'error', 'message': 'Missing image file or model ID'}), 400
    
    image_file = request.files['image_file']
    model_id = request.form['model_id']
    
    if image_file.filename == '':
        return jsonify({'status': 'error', 'message': 'No image file selected'}), 400
    
    # 保存图片
    img_filename = str(uuid.uuid4()) + '.png'
    img_path = os.path.join(app.config['IMAGE_FOLDER'], img_filename)
    image_file.save(img_path)
    
    # 加载图片并预处理
    try:
        # 打开原始图片
        original_img = Image.open(img_path)
        
        # 保存原始图片用于显示
        original_img_path = os.path.join(app.config['IMAGE_FOLDER'], 'original_' + img_filename)
        original_img.save(original_img_path)
        
        # 预处理函数：处理任意尺寸的手写数字图片
        def preprocess_handwritten_digit(image):
            # 转换为灰度图
            if image.mode != 'L':
                image = image.convert('L')
            
            # 提取数字区域
            # 计算非空白像素的边界框
            data = np.array(image)
            non_empty_columns = np.where(data.min(axis=0) < 200)[0]
            non_empty_rows = np.where(data.min(axis=1) < 200)[0]
            
            if len(non_empty_columns) > 0 and len(non_empty_rows) > 0:
                # 有内容的情况
                cropBox = (min(non_empty_columns), min(non_empty_rows), 
                          max(non_empty_columns), max(non_empty_rows))
                
                # 确保裁剪区域有效
                if cropBox[2] > cropBox[0] and cropBox[3] > cropBox[1]:
                    image = image.crop(cropBox)
            
            # 反转颜色（MNIST数据集是黑底白字）
            image = Image.fromarray(255 - np.array(image))
            
            # 添加一点边距
            bordered_width = int(image.width * 1.4)
            bordered_height = int(image.height * 1.4)
            bordered_img = Image.new('L', (bordered_width, bordered_height), 0)
            
            # 将原始图片粘贴到中间位置
            paste_x = (bordered_width - image.width) // 2
            paste_y = (bordered_height - image.height) // 2
            bordered_img.paste(image, (paste_x, paste_y))
            
            # 调整大小到28x28（MNIST标准尺寸）
            image = bordered_img.resize((28, 28), Image.LANCZOS)
            
            # 保存预处理后的图片用于调试
            processed_img_path = os.path.join(app.config['IMAGE_FOLDER'], 'processed_' + img_filename)
            image.save(processed_img_path)
            
            return image
        
        # 应用预处理
        img = preprocess_handwritten_digit(original_img)
        
        # 转换为tensor
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        
        # 加载模型
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_id)
        
        logger.info(f"尝试加载模型: {model_path}")
        
        try:
            # 尝试使用weights_only=False直接加载
            model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            logger.info("成功加载完整模型")
        except Exception as e1:
            logger.error(f"加载完整模型失败: {str(e1)}")
            try:
                # 尝试加载状态字典到预定义模型
                model = LeNet()
                state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
                
                # 检查是否是状态字典或完整模型
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif hasattr(state_dict, 'state_dict'):
                    state_dict = state_dict.state_dict()
                
                model.load_state_dict(state_dict)
                logger.info("成功加载状态字典到LeNet模型")
            except Exception as e2:
                logger.error(f"加载状态字典失败: {str(e2)}")
                # 最后尝试
                try:
                    model = torch.jit.load(model_path, map_location=torch.device('cpu'))
                    logger.info("成功加载TorchScript模型")
                except Exception as e3:
                    logger.error(f"所有加载尝试均失败: {str(e3)}")
                    raise Exception(f"无法加载模型，请确保模型格式兼容。你可以尝试重新训练并使用model.state_dict()保存模型。详细错误: {str(e3)}")
        
        # 设置为评估模式
        model.eval()
        
        # 推理
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(output, dim=1).item()
            class_probabilities = {str(i): float(prob) for i, prob in enumerate(probabilities)}
        
        # 将图片转换为base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 将原始图片转换为base64
        original_buffered = BytesIO()
        original_img.save(original_buffered, format="PNG")
        original_img_str = base64.b64encode(original_buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'predicted_class': predicted_class,
            'probabilities': class_probabilities,
            'image_data': img_str,
            'original_image_data': original_img_str
        })
        
    except Exception as e:
        logger.error(f"推理过程发生错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    eventlet.monkey_patch()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 
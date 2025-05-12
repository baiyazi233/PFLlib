from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import subprocess
import threading
import re
import logging
import eventlet

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variable to store the current training process
current_process = None

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
        
        if current_process.poll() is None:
            logger.info("Terminating training process")
            current_process.terminate()
    
    # Start training in a separate thread
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success'})

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global current_process
    if current_process:
        logger.info("Stopping training process")
        current_process.terminate()
        current_process = None
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    logger.info("Starting Flask application")
    eventlet.monkey_patch()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 
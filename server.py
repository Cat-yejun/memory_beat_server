# server.py
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import base64
import io
from PIL import Image
import numpy as np
import cv2
from threading import Thread
from queue import Queue
from memoryBeat import MemoryBeat

image_queue = Queue()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO(app)

# def draw_circle_on_image(image):
#     # 동그라미의 중심, 반지름, 색상(여기서는 녹색), 두께 설정
#     center_coordinates = (int(image.shape[1] / 2), int(image.shape[0] / 2))
#     radius = 50
#     color = (0, 255, 0)
#     thickness = 2

#     # OpenCV를 사용하여 이미지에 동그라미 그리기
#     image_with_circle = cv2.circle(image, center_coordinates, radius, color, thickness)
#     return image_with_circle

# def display_images():
#     while True:
#         if not image_queue.empty():
#             image = image_queue.get()
#             cv2.imshow('Received Image', image)
#             if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 종료
#                 break
#     cv2.destroyAllWindows()

@app.route('/')
def index():
    # 메인 페이지 렌더링
    return render_template('index.html')

@app.route('/api/combo', methods=['GET'])
def get_combo():
    combo = memoryBeat.get_combo()
    return jsonify({'combo': combo})

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

width, height = 0, 0

@socketio.on('video_frame')
def handle_video_frame(data):
    global width, height
    global image_queue
    # image_data = data['image']  # Base64 인코딩된 이미지 데이터
    # Base64 문자열을 이미지로 변환
    
    width = data.get('width')
    height = data.get('height')
    
    image_queue.put((data, request.sid))
    # print('Received image')
    
    if width is None or height is None:
        print("Width or height is missing.")
        return  # 너비나 높이 정보가 없으면 처리를 중단합니다.
    
    if not processing_thread.is_alive():
        socketio.start_background_task(image_processing_thread)
    # image = base64.b64decode(image_data.split(",")[1])
    # image = np.fromstring(image, np.uint8)
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # image = Image.open(io.BytesIO(image))
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
     # OpenCV를 사용하여 화면에 이미지 띄우기
    
    # processed_image = MemoryBeat.process_frame(image)
    
    # draw_circle_on_image(image)
    
    # retval, buffer = cv2.imencode('.jpg', image)
    # encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    # _, buffer = cv2.imencode('.jpg', processed_image)
    # encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    # 처리된 이미지를 클라이언트에 전송
    # emit('processed_frame', {'image': 'data:image/jpeg;base64,' + encoded_image})
    # image_queue.put(image)
    
    # print('Received image')

def image_processing_thread(memoryBeat):
    global image_queue
    global width, height
    
    while True:
        try:
            data, sid = image_queue.get()
            # print('Processing image')
            image_data = base64.b64decode(data['image'].split(",")[1])
            image = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # print(width, height) 
            processed_image = memoryBeat.process_frame(image, width, height)
            retval, buffer = cv2.imencode('.jpg', processed_image)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            
            socketio.emit('processed_frame', {'image': 'data:image/jpeg;base64,' + encoded_image})
            # print("Processing image for SID:", sid)  # 처리 중인 이미지 로그 출력
        except Exception as e:
            print("An error occurred in the image processing thread:", e)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    # display_thread = Thread(target=display_images)
    # display_thread.daemon = True  # 메인 프로그램 종료시 스레드도 함께 종료되도록 설정
    # display_thread.start()
    print('Starting the server...')
    try:
        memoryBeat = MemoryBeat()

        processing_thread = Thread(target=image_processing_thread, args=(memoryBeat,))
        processing_thread.daemon = True  # 메인 스레드가 종료되면 함께 종료되도록 설정
        processing_thread.start()
    
        socketio.run(app, debug=True, host='0.0.0.0', port=8080, ssl_context=('cert.pem', 'key.pem'))
        
    except Exception as e:
        print(f'An error occurred: {e}')

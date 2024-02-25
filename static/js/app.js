document.addEventListener('DOMContentLoaded', function() {
    var socket = io();
    var video = document.getElementById('video');
    var canvas = document.createElement('canvas');
    var context = canvas.getContext('2d');

    function startCamera() {
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then(function(stream) {
                video.srcObject = stream;
                // 카메라가 준비되면 프레임 전송 시작
                setInterval(sendFrame, 100); // 100ms 마다 프레임 전송
            })
            .catch(function(err) {
                console.error("Error: ", err);
            });
    }

    function sendFrame() {
        if (video.srcObject) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            var dataURL = canvas.toDataURL('image/jpeg', 0.5); // Base64 인코딩
            socket.emit('video_frame', {
                image: dataURL,
                width: video.videoWidth,
                height: video.videoHeight
            });
        }
    }
    

    startCamera();
    
    // 서버로부터 처리된 이미지 데이터를 수신하는 이벤트 리스너 추가
    socket.on('processed_frame', function(data) {
        var frame = document.getElementById('frame');
        if (frame) {
            frame.src = data.image; // 받은 이미지 데이터로 img 태그의 src 속성 업데이트
        }
    });
});

from flask import Flask, Response, request, render_template
import zmq
import cv2
import numpy as np

app = Flask(__name__)


# 视频监控
@app.route('/monitor', methods=('GET', 'POST'))
def monitor():
    return render_template('monitor.html')



@app.route('/video_monitor')
def video_monitor():
    return Response(video_monitor_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


def video_monitor_gen():
    context = zmq.Context()
    # zmq.REQ-->客户端：先send,再recv
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:11556")

    while True:
        # 客户端，先send，再recv
        socket.send(b"a")  # b表示二进制
        data = socket.recv()

        # 将接受到的二进制图片，通过cv2.imdecode解码为image
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), 1)
        # # 可视化
        # cv2.imshow("image", image)
        # # key = cv2.waitKey(0)
        # key = cv2.waitKey(10)
        # if key & 0xFF == ord('q'):
        #     break
        show_frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + show_frame + b'\r\n')



if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, debug=True)
    # app.run(host="192.168.16.109", port=54321, debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
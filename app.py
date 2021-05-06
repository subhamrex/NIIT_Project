from flask import Flask, render_template, Response
from mask_detect_video_app import VideoCamera
from flask import send_file

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/project')
def project():
    """ Project Page"""
    return render_template('project.html')

@app.route('/about_me')
def about_me():
    """ About Me page"""
    return render_template('about_me.html')

@app.route('/download')
def downloadFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    path = "NIIT_Project.pdf"
    return send_file(path, as_attachment=True)    


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')
                 
if __name__ == "__main__":
    app.run(debug=True)
                 
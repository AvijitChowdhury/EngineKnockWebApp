import streamlit as st
import pandas as pd
from io import StringIO
import moviepy.editor as mp
import librosa
import matplotlib.pyplot as plt
import librosa.display
from PIL import ImageMath
from fastai.data.external import *
from fastai.vision.all import *

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
st.title('Engine Knocking Predictor')
uploaded_video = st.file_uploader("Choose a file",type = ['mp4','mpeg','mov'])
try:
    uploaded_video.name="random.mp4"
except:
    print("")

if uploaded_video is not None:
    
     
        
    vid = uploaded_video.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read()) # save video to disk
        st_video = open(vid,'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
    
if st.button('Predict'):
    learn = load_learner("Model/model_v2.pkl")
    mov_filename = 'random.mp4'
    project_path = '.'

    clip = mp.VideoFileClip(project_path + '/' + mov_filename)
    clip_start = (clip.duration/2)-1
    clip_end = (clip.duration/2)+1
    clip = clip.subclip(clip_start,clip_end)
    sr = clip.audio.fps
    y = clip.audio.to_soundarray()
    y = y[...,0:1:1].flatten()

    D = librosa.stft(y)
    D_harmonic, D_percussive = librosa.decompose.hpss(D)
    rp = np.max(np.abs(D))

    side_px=256
    dpi=150
    plot = plt.figure(figsize=((side_px/dpi), (side_px/dpi)))

    CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(librosa.istft(D_percussive), sr=sr)), ref=np.max)
    p=librosa.display.specshow(CQT,x_axis=None,y_axis=None)
    plt.axis('off')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.pyplot(plot.canvas.draw())

    im_data = np.fromstring(plot.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    im_data = im_data.reshape(plot.canvas.get_width_height()[::-1] + (3,))

    pred_class,pred_idx,outputs = learn.predict(im_data)
    st.success(pred_class)
    st.write('Accuracy Score: ')
    st.write('Knocking: {} % '.format((outputs[0])*100))
    st.write('Normal: {} % '.format((outputs[1])*100)) 
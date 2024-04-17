'''
Date: 2024-03-27 21:48:51
LastEditors: Haku nishimiyahaku@outlook.com
LastEditTime: 2024-04-17 21:55:57
FilePath: \test\run_eval.py
Description:  pass
'''
import pandas as pd
import os
from vf_model import vf_model
import pickle
import argparse
import numpy as np
import pyaudio, wave, librosa
from scipy.io import wavfile
import tensorflow as tf

class eval():
    def __init__(self, name, csvPath, dataset_path, savePath):
        self.Name = name
        self.dataset_path = dataset_path
        self.csvPath = csvPath
        self.savePath = savePath
        self.map = {0: "teens", 1: "adults", 2: "midlife/old"}
        self.model_class = vf_model(Name, csvPath, dataset_path)
        self.model_class.build_vf_model()
        self.model_class.model.load_weights(os.path.join(savePath, 'model.h5'))


    def process_file(self, filename):

        spec_infos=[]
        spec_infos.append(self.model_class.get_mel(filename))
        mels_strength = np.array([s[1] for s in spec_infos])

        scaler = pickle.load(open('./models_'+ "test"+ '/'+ 'scale.pkl', 'rb'))
        x = scaler.transform(mels_strength)
        # x = tf.nn.l2_normalize(mels_strength,axis=1)

        predicted = self.model_class.model.predict(x, verbose=0)
        return predicted

    def process_realtime(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024)
        while True:  
            print("开始录音...")
            frames = []
            for i in range(0, int(16000 / 1024 * 3)):
                data = stream.read(1024)
                frames.append(data)         
            wf = wave.open("1.wav", 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(frames))
            wf.close()
            print("录音结束")
            result = self.process_file('1.wav')
            print(result)
            print(self.map.get(np.argmax(result[0]), "invalid"), 
                  "female" if result[1] >= 0.5 else "male  ", 
                  end='\r', flush=True)


if __name__ == '__main__':
    Name = "test"
    dataset_path = "./dataset/cv-valid-test/"
    csvPath = "./dataset/cv-valid-test.csv"
    savePath = './models_' + Name + '/'
    eval_class = eval(Name, csvPath, dataset_path, savePath)
    eval_class.process_realtime()

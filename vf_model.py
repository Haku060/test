import os
import warnings
warnings.filterwarnings('ignore')
import pickle
import librosa
import numpy as np
import pandas as pd
import librosa.display
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm



class vf_model():
    def __init__(self, name, csvPath, dataset_path):
        self.sr = 16000
        self.model = []
        self.Name = name
        self.csvPath = csvPath
        self.dataset_path = dataset_path
        self.savePath = './models_' + self.Name + '/' 
        self.epochs = 512
        self.batch_size = 128
        self.validation_split = 0.2
        self.StandardScaler = StandardScaler()
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)


    def resize_spectrogram(self, spec, length, fact=-80):
        canvas = np.ones((len(spec), length)) * fact
        if spec.shape[1] <= length:
            canvas[:, : spec.shape[1]] = spec
        else:
            canvas[:, :length] = spec[:, :length]
        return canvas


    def get_mel(self, filename, sr=16_000, hop_length=2048, duration=3.0):
        y, sr = librosa.load(filename, sr=sr)
        x_mel = librosa.feature.melspectrogram(y=y, sr=sr)
        x_mel = librosa.power_to_db(x_mel, ref = np.max)
        mel_strength = np.nanmean(x_mel, axis=1)
        length = int(duration * sr / hop_length)
        x_mel = self.resize_spectrogram(x_mel, length, fact=-80)
        return x_mel, mel_strength


    def dataset_generator(self):
        print("////dataset_generating////")
        spec_infos = []
        df = pd.read_csv(self.csvPath)
        df.loc[:, "age"] = df["age"].map({
            "teesns":    0, 
            "twenties": 1, "thirties":   1, "fourties":  1, "fifties":  1,
            "sixties":  2, "senventies": 2, "eighties":  2})
        df.loc[:, "gender"] = df["gender"].map({"male": 0, "female": 1})
        df.dropna(subset=['age','gender'], inplace=True)

        y_age = df["age"].values
        y_gender = df["gender"].values
        y_age = np.array(y_age, dtype=np.int32)
        y_gender = np.array(y_gender, dtype=np.int32)

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
            filenames = row[0]
            if os.path.isfile(os.path.join(self.dataset_path,filenames)):
                wav_file_path = os.path.join(self.dataset_path,filenames)
                spec_infos.append(self.get_mel(wav_file_path, sr=self.sr))
        # mels = np.array([s[0] for s in spec_infos])
        mels_strengths = np.array([s[1] for s in spec_infos])

        input_feature = self.StandardScaler.fit_transform(mels_strengths)
        pickle.dump(self.StandardScaler, open(self.savePath + 'scale.pkl','wb'))
        np.savez(os.path.join(self.savePath, 'features.npz'), 
                 input_feature=input_feature, 
                 y_age=y_age, 
                 y_gender=y_gender)



    def build_vf_model(self):
        inputs = keras.layers.Input(shape=[128,])
        x1 = keras.layers.Dense(128, activation='relu')(inputs)
        x1 = keras.layers.Dense(1024, activation='relu')(x1)
        x1 = keras.layers.Dense(2048, activation='relu')(x1)
        out1 = keras.layers.Dense(3, activation='softmax')(x1)
        
        x2 = keras.layers.Dense(128, activation='relu')(inputs)
        x2 = keras.layers.Dense(512, activation='relu')(x2)

        out2 = keras.layers.Dense(1, activation='sigmoid')(x2)

        self.model = keras.models.Model(inputs=inputs, outputs=[out1,out2])
        print(self.model.summary())



    def compile_model(self):
        self.model.compile(
            optimizer = "adam",
            loss=[keras.losses.SparseCategoricalCrossentropy(),
                  keras.losses.BinaryCrossentropy()],
            # loss = keras.losses.BinaryCrossentropy(),
            metrics=['accuracy'])
    

    def train_model(self):
        try:
            dataset = np.load(os.path.join(self.savePath, 'features.npz'), allow_pickle=True)
        except:
            raise Exception("dataset.npz not found, please run 'gen_dataset.py' to create dataset")

        features = dataset['input_feature']
        y_age = dataset['y_age']
        y_gender = dataset['y_gender']

        features = tf.nn.l2_normalize(features,axis=1)

        csv_logger = keras.callbacks.CSVLogger(self.savePath+ 'training_' + self.Name + '.log')

        early_stopping = keras.callbacks.EarlyStopping(monitor = 'loss',
                                                       patience = 10,
                                                       restore_best_weights = True)
        checkpointer = keras.callbacks.ModelCheckpoint(self.savePath + 'model.h5',
                                                       monitor='val_loss',
                                                       verbose = 1,
                                                       save_best_only=True,
                                                       save_weights_only=True,
                                                       mode='auto',
                                                       save_freq='epoch')
        tf.random.set_seed(1)   


        self.model.fit(
            features,
            [y_age,y_gender],
            epochs = self.epochs,
            validation_split = self.validation_split,
            batch_size = self.batch_size,
            callbacks=[csv_logger, early_stopping, checkpointer])

        keras.backend.clear_session()



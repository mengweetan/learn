import tensorflow as tf
import numpy as np
import pandas as pd
import os, datetime


Model = tf.keras.models.Model
Sequential =  tf.keras.models.Sequential
Input = tf.keras.layers.Input
LSTM = tf.keras.layers.LSTM
Embedding = tf.keras.layers.Embedding
Dense = tf.keras.layers.Dense
concatenate = tf.keras.layers.concatenate
Flatten = tf.keras.layers.Flatten
Add =  tf.keras.layers.Add
Constant = tf.keras.initializers.Constant

load_model = tf.keras.models.load_model

to_categorical = tf.keras.utils.to_categorical

KerasClassifier = tf.keras.wrappers.scikit_learn.KerasClassifier

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split



class Machine:
    def __init__(self, ):
        
        # self.HAIKU_LINES_NUM = 3
        file_path = os.getcwd()+'/'
        
        self.dataDir=file_path +''
        #self.df = pd.read_csv(self.dataDir+'__INPUT.txt', sep = '\t')
        self.df = pd.read_csv(self.dataDir+'_final.csv')
        #self.size = size if size else self.df.shape[0]

        #self.build_matrix = build_matrix # no need for now
        self._setup()
        
    
    def _setup(self):
        
        
        _dataset = self.df.drop(['__temp'],axis=1)
        _num_columns = _dataset.shape[1]
        dataset = _dataset.values




        self.X = dataset[:,0:_num_columns-1].astype(float)
        _y = dataset[:,_num_columns-1]  # the last one is always the label
        print (self.X.shape)
        
        encoder = LabelEncoder()
        encoder.fit(_y)
        encoded_Y = encoder.transform(_y)

        #print ('num of y'.format(len(set(encoded_Y))))
        self.y = to_categorical(encoded_Y)
        print (self.y.shape)
        model = Sequential()
        model.add(Dense(15, input_dim=_num_columns-1, activation='relu'))
        model.add(Dense(self.y.shape[1], activation='softmax')) # because only 4 labels
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print (model.summary())
        self.mymodel = model
        
    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        history = self.mymodel.fit(X_train,y_train, validation_data=(X_test,y_test), shuffle=True, epochs=150 )

        #history = model.fit_generator(training_generator, validation_data=validation_generator,  epochs=epochs, use_multiprocessing=True, callbacks=[mc,es,tfc,lrc])


        self.mymodel.save(self.dataDir+'/_modelv2.h5')
        print ('saved model in {}'.format(self.dataDir))
        print (history)
        
        #estimator = KerasClassifier(build_fn=self.model, epochs=200, batch_size=5, verbose=0)
        #kfold = KFold(n_splits=10, shuffle=True)
        #results = cross_val_score(estimator, self.X, self.dummy_y , cv=kfold)
        
    def infer(self):
        model.predict([data])
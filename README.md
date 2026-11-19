### AIM : 







Train a small feedforward neural network to classify non-linear data. 








### ALGORITHM : 









Two-layer dense network with ReLU hidden units and sigmoid output, trained with Adam. 








### THEORY:







With non-linear activations, networks approximate complex decision boundaries 
(universal approximation).








### INPUT : 



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from sklearn.datasets import make_moons

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


x,y = make_moons(n_samples=500, noise=0.2, random_state=42)

xtr,xte,ytr,yte = train_test_split(x,y,test_size=0.2,random_state=42)

model = Sequential([Dense(8, activation='relu', input_shape=(2,)), Dense(1, activation='sigmoid')])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(xtr, ytr, epochs=30, batch_size=16, verbose=0)

y_pred = (model.predict(xte) >0.5).astype(int).flatten()

print('NN test accuracy:', round(accuracy_score(yte, y_pred),4))






### OUTPUT :


/usr/local/lib/python3.13/dist-packages/keras/src/layers/core/dense.py:106: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
  
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step

NN test accuracy: 0.87










### RESULT : 







Neural networks model non-linear boundaries; students see improved accuracy over 
linear methods. 

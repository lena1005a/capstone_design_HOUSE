import numpy as np

gesture = ['ALT_TAB', 'ALT_F4', 'FULL', 'SOUND_CONTROL']

data = np.concatenate([

], axis=0)

x_data = data[:, :, :-1] 
labels = data[:, 0, -1]

from tensorflow.keras.utils import to_categorical
y_data = to_categorical(labels, num_classes=len(gesture))

from sklearn.model_selection import train_test_split
x_test = x_data.astype(np.float32)
y_test = y_data.astype(np.float32)


print("최종")
print(x_test.shape, y_test.shape) 

from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras.models import load_model
model = load_model('models/cursor_model_t1.h5')
y_pred = model.predict(x_test)
test_p = multilabel_confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

for i in range(len(test_p)):
    print(gesture[i],"\n", test_p[i], "\n")
    
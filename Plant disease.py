import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
data_dir = "E:/Programs/Plant/PlantVillage"
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
val = train_datagen.flow_from_directory(
    data_dir,target_size=(224, 224),batch_size=32,class_mode='categorical',subset='validation')
class_names = list(train.class_indices.keys())
num_classes = len(class_names)
model_filename = "plant_disease_model.keras"
if os.path.exists(model_filename):
    print("Model already exists. Loading the model...")
    model = load_model(model_filename)
    model.save("E:/Programs/Plant/plant_disease_model.keras")
else:
    print("Training new model...")
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),MaxPooling2D(2,2),Conv2D(64, (3,3), activation='relu'),MaxPooling2D(2,2),Conv2D(128, (3,3), activation='relu'),MaxPooling2D(2,2),Flatten(),Dropout(0.5),Dense(128, activation='relu'),Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(train, validation_data=val, epochs=5)
    model.save(model_filename)
img_path= r"E:\Programs\Plant\test_leaf.jpg"  
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0  
img_array = np.expand_dims(img_array, axis=0)  
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)
plt.imshow(img)
plt.title(f"{predicted_class} ({confidence:.2f})")
plt.axis('off')
plt.show()
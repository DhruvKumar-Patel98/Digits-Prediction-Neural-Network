from tensorflow import keras

(X_train, Y_train) , (X_test, Y_test) = keras.datasets.mnist.load_data()
X_train = X_train/255
X_test = X_test/255
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),  # First layer
    keras.layers.Dense(256, activation='relu'),    
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train, Y_train, epochs=5)
model.evaluate(X_test, Y_test)
model.save('handwritten.keras')
# model.export('C:\\Users\\Dhruvkumar Patel\\OneDrive\\Desktop\\SEMO\\Adv-AI\\Handwritting-detactor\\DeepLearningPython')



# y_predicted = model.predict(X_test)
# y_predicted[0]
# print(np.argmax(y_predicted[0]))
# plt.imshow(X_test[0])

# loss, accuracy = model.evaluate(x_test,y_test)
# print(loss)
# print(accuracy)
# image = x_test[0]  # This is your (28, 28) image

# # Reshape the image to add a batch dimension
# image = np.expand_dims(image, axis=0)  
# prediction = model.predict(image)
# plt.imshow(x_test[0],cmap=plt.cm.binary)
# plt.show()
# print(np.argmax(prediction))
# res = model.predict(x_test[0])

# print(res)

# image_number=1
# while os.path.isfile(f"digits/digit{image_number}.png"):
#     img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
#     img = np.array([img])
#     prediction = model.predict(img)
#     print(f"This digits is probably a {np.argmax(prediction)}")
#     plt.imshow(img[0],cmap=plt.cm.binary)
#     plt.show()
#     image_number+=1

# for i in range (0,10):
#     image = x_test[i] 
#     image = np.expand_dims(image, axis=0)  
#     prediction = model.predict(image)
#     plt.imshow(x_test[i],cmap=plt.cm.binary)
#     print("This digits is probably a: ", np.argmax(prediction))
#     plt.show()

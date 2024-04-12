from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

"""
This class defines our CNN (ResNet50) architecture
"""
class ModelHandler:
    def __init__(self):
        pass
    
    def define_model(self):
        base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = True
            
        base_model_output = base_model.output
        
        x = GlobalAveragePooling2D()(base_model_output)
        x = Dense(512, activation='relu')(x)
        x = Dense(5, activation='softmax', name='fcnew')(x)
        model = Model(base_model.input, x)
        
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', 'auc'])
        return model
    
    def train_model(self, model, train_data, validation_data):
        history = model.fit(train_data, epochs=10, validation_data=validation_data)
        return history


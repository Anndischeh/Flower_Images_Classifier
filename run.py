from preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from model import ModelHandler
from plotting import ShowResult
from pathlib import Path
import pandas as pd
import os

# Load data
image_dir = Path('/kaggle/input/flowers-recognition/flowers')
file_paths = list(image_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], file_paths))
file_paths = pd.Series(file_paths, name="FilePath").astype(str)
labels = pd.Series(labels, name='Labels')
df = pd.concat([file_paths, labels], axis=1)
df = df.sample(frac=1).reset_index(drop=True)

# Split data into train and test
train_df, test_df = train_test_split(df, train_size=0.9, shuffle=True, random_state=1)

# Initialize classes
preprocessor = Preprocessing()
model_handler = ModelHandler()
result_shower = ShowResult()

# Preprocessing
images = plt.imread(df.FilePath[15])
norm1, norm2, norm3 = preprocessor.normalize_image(images)
transformed_images = preprocessor.apply_transformations(images)
train_images, validation_images, test_images = preprocessor.create_image_generators(train_df, test_df)

# Define model
model = model_handler.define_model()

# Train model
history = model_handler.train_model(model, train_images, validation_images)

# Visualize results
result_shower.visualize_training_results(history)

# Evaluate model
test_results = model.evaluate(test_images)
print("Test Accuracy:", round(test_results[1], 3))
print("Test AUC:", round(test_results[2], 3))

# Display predictions
predictions = model.predict(test_images)
result_shower.display_predictions(test_df, predictions)

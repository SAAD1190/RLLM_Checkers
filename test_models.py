from keras.models import load_model
import os
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

# Get the current working directory
path = os.getcwd()
print(f"Current working directory: {path}")

# Path to the models folder
models_folder = os.path.join(path, "models")

# Check if the models folder exists
if os.path.exists(models_folder):
    model_files = os.listdir(models_folder)
    print(f"Model files in the 'models' directory: {model_files}")
    
    # Iterate through each model file and attempt to load it
    for model_file in model_files:
        model_path = os.path.join(models_folder, model_file)
        print(f"\nAttempting to load model: {model_file}")
        
        try:
            # Load the model
            model = load_model(model_path)
            print(f"Model '{model_file}' loaded successfully!")
        except Exception as e:
            print(f"Error loading model '{model_file}': {e}")
else:
    print(f"Models folder not found at: {models_folder}")
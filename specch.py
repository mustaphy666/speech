import streamlit as st
st.title("Speech Command Recognition")
st.write("This app recognizes speech commands using a pre-trained model.")
import torch
import librosa
import numpy as np
import os
import torch.nn as nn

class SpeechCommandCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),  
            nn.ReLU(),
            nn.MaxPool2d(2),                
            nn.Conv2d(16, 32, kernel_size=3), 
            nn.ReLU(),
            nn.MaxPool2d(2),                 
        )
        dummy_input = torch.zeros(1, 1, 40, 98)  
        with torch.no_grad():
            dummy_output = self.conv(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SpeechCommandCNN(num_classes=6)
model.load_state_dict(torch.load("speech_command_mfcc.pth", map_location="cpu"))
model.eval()

def preprocess_audio_librosa(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 98:
        pad_width = 98 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :98]
    
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,40,98)
    return mfcc
def predict_command(mfcc):
    with torch.no_grad():
        output = model(mfcc)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)
    return predicted_index.item(), confidence.item()
st.write("Upload an audio file:")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])
if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio("temp_audio.wav")
    mfcc = preprocess_audio_librosa("temp_audio.wav")
    command_index, confidence = predict_command(mfcc)
    commands = ['yes', 'no', 'up', 'down', 'left', 'right']
    predicted_command = commands[command_index] if command_index < len(commands) else "Unknown"
    
    st.write(f"Predicted Command: {predicted_command} (Confidence: {confidence:.2f})")
    os.remove("temp_audio.wav") 
else:
    st.write("Please upload an audio file to make a prediction.") 


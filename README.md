# 🎙️ Speech Command Recognition System

> **A deep learning CNN trained to classify spoken commands in real time — achieving 85% validation accuracy on the Google Speech Commands dataset.**



---

## 🧠 What It Does

This system listens to short spoken commands and classifies them into one of several categories in real time. It demonstrates how audio signals can be transformed into visual representations (spectrograms) and fed into a CNN — the same technique used in voice assistants and smart devices.

**Supported commands:**
`yes` · `no` · `up` · `down` · `left` · `right` · `on` · `off` · `stop` · `go`

---

## 🔍 How It Works

Raw audio is not directly fed into the CNN. Instead it goes through a transformation pipeline:

```
Raw Audio (.wav)
      │
      ▼
Mel Spectrogram
(audio → image representation)
      │
      ▼
CNN Classifier
(treats spectrogram like an image)
      │
      ▼
Predicted Command + Confidence Score
```

Converting audio to a **Mel Spectrogram** is the key insight — it turns a 1D audio signal into a 2D image that a CNN can process with the same power it uses for visual recognition.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Validation Accuracy | **85%** |
| Architecture | Deep CNN (PyTorch) |
| Dataset | Google Speech Commands |
| Input Format | Mel Spectrogram |
| Commands | 10 classes |

---

## 🏗️ Project Structure

```
speech-command-recognition/
├── data/                        # Google Speech Commands dataset
├── notebooks/
│   └── training.ipynb           # EDA, preprocessing, training
├── model/
│   ├── train.py                 # Training pipeline
│   ├── cnn_model.py             # CNN architecture
│   └── model.pth                # Saved model weights
├── utils/
│   └── audio_processing.py      # Mel spectrogram transformation
├── app.py                       # Streamlit web app (real-time input)
├── requirements.txt
└── README.md
```

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Audio Processing | Librosa, torchaudio |
| Model Architecture | CNN |
| Web App | Streamlit |
| Visualisation | Matplotlib |
| Language | Python 3.11+ |

---

## 🛠️ Setup

```bash
git clone https://github.com/mustaphy666/speech-command-recognition.git
cd speech-command-recognition
pip install -r requirements.txt
streamlit run app.py
```

**Note:** Microphone access is required for real-time recognition mode.

---

## 📁 Dataset

Uses the [Google Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) dataset — a collection of 65,000 one-second audio clips of 30 short words.

---

## 👤 Author

**Saheed Mustapha Olatunji**
- GitHub: [@mustaphy666](https://github.com/mustaphy666)
- LinkedIn: [mustapha-saheed](https://www.linkedin.com/in/mustapha-saheed)

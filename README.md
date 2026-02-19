# 🎙️ Voice Timbre Transfer (RVC)

A high-performance vocal conversion application powered by Retrieval-based Voice Conversion (RVC). Transform vocal tracks into target voices with high fidelity.

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.11+** is recommended.
- macOS (Apple Silicon supported) or Linux.

### 2. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Load Voice Models
- Place your RVC target voice models (`.pth` files) in the `voices/` directory.
- The app automatically discovers models on startup.

### 4. Running the App
Launch the Streamlit frontend:
```bash
streamlit run app.py
```
*On start, the app will automatically download necessary backbone weights (`hubert_base.pt`, `rmvpe.pt`) if they are missing.*

## 📂 Project Structure
- `voices/`: Drop your `.pth` voice models here.
- `output/`: Converted audio files are saved here.
- `assets/`: Backbone weights (HuBERT/RMVPE) are stored here.
- `engine.py`: Core inference logic and compatibility patches.
- `app.py`: Streamlit-based UI with caching and resource management.

## ⚡ Key Features & Optimizations
- **Energy Efficient**: Optimized startup using marker files to avoid redundant system-wide patching.
- **Fast UI**: Cached audio analysis prevents CPU spikes during parameter adjustments.
- **Resource Management**: Includes a sidebar button to **Unload Model & Clear RAM**, essential for freeing up memory/GPU when idle.
- **Auto-Scale**: Detects and uses **MPS (Metal Performance Shaders)** on Apple Silicon for accelerated inference.

## 🎧 Usage
1. Upload a vocal track (.wav or .mp3).
2. Select a target voice from the sidebar.
3. Adjust **Pitch Transpose** (+12 for Male → Female conversion).
4. Click **Convert** and download the result.

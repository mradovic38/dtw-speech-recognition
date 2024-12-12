# Dynamic Time Warping (DTW) Speech Recognition

A speech recognition system that uses feature extraction and dynamic time warping (DTW) to identify words and to find the most similar speaker.

## Features
* **Word Recognition**: Identifies words spoken by different speakers using a weighted voting mechanism across multiple feature types.
* **Speaker Similarity Detection**: Determines the most similar speaker to a target speaker based on feature comparisons.
* **Dynamic Time Warping (DTW)**: Measures similarity between time-series data with a focus on minimizing cost paths.
* **Multiple Feature Types**: Supports MFCC, LPC, and Mel Spectrogram features for diverse and accurate audio analysis.
* **Confidence and Threshold Handling**: Ensures reliable predictions by applying confidence thresholds and cost checks.

## Setup
1. Clone the repository:
  ```bash
  git clone https://github.com/mradovic38/dtw-speech-recognition
  cd dtw-speech-recognition
  ```
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage
1. Create a directory of audio files of different speakers saying the same group of words.
2. Name your audio files as `word-speaker.wav` (ex. `down-mark.wav`).
4. For details on how to run the project on your data, refer to the [run.py](run.py) file. It contains examples and explanations for word recognition and speaker similarity detection.

## Customization
* Adjust feature weights in [word_recognition.py](word_recognition.py) to prioritize specific features.
* Modify confidence and cost thresholds to suit different datasets.

## How it works
### Feature Extraction:
Extracts audio features such as MFCC, LPC, and Mel spectrograms from input audio files.
### Dynamic Time Warping (DTW):
Computes similarity between time-series data (input audio vs. database).
### Word Recognition: 
Aggregates feature-based predictions using weighted voting across all three feature types 
for reliable word identification.
### Speaker Similarity: 
Compares speakers by averaging DTW costs across shared vocabulary based on a passed feature
extraction alghoritm.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

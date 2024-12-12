
import librosa
import numpy as np

def extract_features(file_path, feature_type="mel"):
    """
    Extract audio features from a file.
    
    Args:
        file_path (str): Path to the audio file
        feature_type (str): Type of feature extraction ('mfcc', 'mel', or 'lpc')
    
    Returns:
        numpy.ndarray: Extracted features
    """
    y, sr = librosa.load(file_path, sr=48000)
    y[y == 0] = 1e-10
    y = librosa.util.normalize(y)

    if feature_type == "mfcc":
        # Improved MFCC extraction
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        features = mfccs.T
    
    elif feature_type == "mel":
        # Enhanced Mel spectrogram extraction
        n_fft = 2048
        hop_length = 512
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, 
            n_fft=n_fft, hop_length=hop_length
        )
        # Log-mel with normalization
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
        # Normalize the Mel spectrogram
        log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / np.std(log_mel_spec)

        features = log_mel_spec.T
    
    elif feature_type == "lpc":
        # Robust LPC feature extraction
        lpc_order = 10
        frame_length = 2048
        hop_length = 512
        
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        
        lpc_features = []
        for frame in frames.T:
            lpc_coeffs = librosa.lpc(frame, order=lpc_order)
            lpc_features.append(lpc_coeffs[1:])
        
        features = np.array(lpc_features)
    
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    
    return features
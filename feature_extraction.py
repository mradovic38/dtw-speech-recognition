
import librosa
import numpy as np

def extract_features(file_path: str, feature_type:str="mel") -> np.ndarray:
    """
    Extract audio features from a file.
    
    Args:
        file_path (str): Path to the audio file
        feature_type (str): Type of feature extraction ('mfcc', 'mel', or 'lpc')
    
    Returns:
        numpy.ndarray: Extracted features
    """
    y, sr = librosa.load(file_path, sr=22000)
    y[y == 0] = 1e-10
    y = librosa.util.normalize(y)

    if feature_type == "mfcc":
        # MFCC extraction with normalization
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        features = mfccs.T
    
    elif feature_type == "mel":
        # Mel spectrogram extraction with normalization
        n_fft = 2048
        hop_length = 512
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=30, 
            n_fft=n_fft, hop_length=hop_length
        )
        # Log-mel with normalization
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
        # Normalize the Mel spectrogram
        log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / np.std(log_mel_spec)

        features = log_mel_spec.T
    
    elif feature_type == "lpc":
        # LPC feature extraction with normalization
        lpc_order = 10
        frame_length = 2048
        hop_length = 512
        
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        
        lpc_features = []
        for frame in frames.T:
            lpc_coeffs = librosa.lpc(frame, order=lpc_order)
            lpc_features.append(lpc_coeffs[1:])

        lpc_features = (lpc_features - np.mean(lpc_features)) / np.std(lpc_features)
        
        features = np.array(lpc_features)
    
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    
    return features
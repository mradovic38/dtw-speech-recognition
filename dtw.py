import numpy as np
import os
import librosa
import scipy.spatial.distance as dist

from utils import load_speakers_from_folder

def dp(dist_mat):
    """
    Find minimum-cost path through matrix using dynamic programming.
    Returns path indices and cost matrix.
    """
    N, M = dist_mat.shape
    
    # Initialize cost matrix with infinity
    cost_mat = np.zeros((N + 1, M + 1))
    cost_mat[1:, 0] = np.inf
    cost_mat[0, 1:] = np.inf
    
    # Fill cost matrix with traceback information
    traceback_mat = np.zeros((N, M), dtype=int)
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]   # deletion (2)
            ]
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty
    
    # Traceback from bottom right
    i, j = N - 1, M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        # Match
        if tb_type == 0:
            i, j = i - 1, j - 1
        # Insertion
        elif tb_type == 1:
            i = i - 1
        # Deletion
        elif tb_type == 2:
            j = j - 1
        path.append((i, j))
    
    # Strip infinity edges from cost matrix
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)

def extract_features(file_path, feature_type="mel"):
    """
    Extract audio features from a file.
    
    Args:
        file_path (str): Path to the audio file
        feature_type (str): Type of feature extraction ('mfcc', 'mel', or 'lpc')
    
    Returns:
        numpy.ndarray: Extracted features
    """
    y, sr = librosa.load(file_path, sr=44000)
    y[y == 0] = 1e-10

    if feature_type == "mfcc":
        # Improved MFCC extraction
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5, n_fft=2048, hop_length=512)
        features = mfccs.T
    
    elif feature_type == "mel":
        # Enhanced Mel spectrogram extraction
        n_fft = 2048
        hop_length = 512
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=40, 
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

def dtw_cost(seq1, seq2):
    """Calculate DTW distance using cosine distance and dynamic programming"""
    dist_mat = dist.cdist(seq1, seq2, "cosine")
    _, cost_mat = dp(dist_mat)
    
    # Normalize alignment cost
    normalized_cost = cost_mat[-1, -1] / (seq1.shape[0] + seq2.shape[0])
    return normalized_cost

def recognize_word(input_file, sound_database, mel_cost_threshold=0.2, confidence_threshold=0.8, input_speaker_name=''):
    """
    Recognize word with improved scoring mechanism:
    - Calculate mean DTW cost for each word across all speakers for each feature type.
    - Predict the best word for each feature type.
    - Use weighted voting based on these predictions to determine the final word.
    - Check if MFCC's best mean value exceeds a threshold.
    """
    feature_types = ['mfcc', 'lpc', 'mel']
    weights = {'mfcc': 2, 'mel': 2, 'lpc': 1}

    # Load speaker files and initialize variables
    speaker_files = load_speakers_from_folder(sound_database)
    vocab = np.unique([x.split('-')[0] for x in os.listdir(sound_database)])
    word_costs = {feature: {word: [] for word in vocab} for feature in feature_types}

    # Calculate DTW costs for each feature type
    for feature_type in feature_types:
        print(f"Extracting {feature_type.upper()} features...")
        input_features = extract_features(input_file, feature_type)

        for speaker in speaker_files:
            if input_speaker_name == speaker:
                continue
            
            print(f'\tSpeaker: {speaker}')
            for word in speaker_files[speaker]:
                if word not in vocab:
                    continue
                
                file_path = speaker_files[speaker][word]
                ref_features = extract_features(file_path, feature_type)
                cost = dtw_cost(input_features, ref_features)
                word_costs[feature_type][word].append(cost)
                print(f'\t\tWord: {word}, cost: {cost}')

    # Calculate mean costs per word for each feature type
    mean_costs = {feature: {} for feature in feature_types}
    for feature_type in feature_types:
        for word in vocab:
            costs = word_costs[feature_type][word]
            mean_costs[feature_type][word] = np.mean(costs) if costs else np.inf

    # Generate predictions for each feature type
    predictions = {}
    feature_votes = []
    for feature_type in feature_types:
        best_word = min(mean_costs[feature_type], key=mean_costs[feature_type].get)
        best_cost = mean_costs[feature_type][best_word]
        print(f"{feature_type.upper()} prediction: {best_word} with mean cost {best_cost}")
        predictions[feature_type] = best_word
        feature_votes.append(feature_type)

    # Weighted voting
    word_scores = {word: 0 for word in vocab}
    for feature_type, predicted_word in predictions.items():
        if predicted_word:
            word_scores[predicted_word] += weights[feature_type]

    # Final word selection based on scores
    print(f"Word scores: {word_scores}")
    final_word = max(word_scores, key=word_scores.get) 
    total_score = sum(word_scores.values())
    confidence = word_scores[final_word] / total_score if total_score > 0 else 0

    print(f"Final word: {final_word}, Confidence: {confidence:.2f}")

    # Confidence thresholding
    if confidence < confidence_threshold:
        print("Confidence below threshold. Classifying as unknown.")
        return None, final_word

    # Check if Mel cost exceeds the threshold
    mel_best_word = predictions['mel']
    if mel_best_word:
        mel_best_cost = mean_costs['mel'][mel_best_word]
        if mel_best_cost > mel_cost_threshold:
            print("Mel Spec. best cost exceeds threshold. Classifying as unknown.")
            return None, final_word
            

    return final_word, final_word


# Step 4: Find Similar Speaker
def find_similar_speaker(input_file, input_speaker_name, speaker_files, feature_type="mfcc"):
    input_features = extract_features(input_file, input_speaker_name, feature_type)

    closest_speaker = None
    min_distance = float('inf')

    for speaker, files in speaker_files.items():
        distances = []
        for file_path in files:
            speaker_features = extract_features(file_path, input_speaker_name, feature_type)
            distances.append(dtw_distance(input_features, speaker_features))
        avg_distance = np.mean(distances)
        print(f"Average DTW distance for speaker '{speaker}' is {avg_distance:.2f}")
        if avg_distance < min_distance:
            min_distance = avg_distance
            closest_speaker = speaker

    return closest_speaker, min_distance

import numpy as np
import librosa
import scipy.spatial.distance as dist
from scipy.stats import zscore

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
        if tb_type == 0:
            i, j = i - 1, j - 1
        elif tb_type == 1:
            i = i - 1
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
    y, sr = librosa.load(file_path, sr=None)
    
    if feature_type == "mfcc":
        # Improved MFCC extraction
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5, n_fft=2048, hop_length=512)
        # Normalize and transpose
        mfccs = (mfccs - mfccs.mean(axis=1, keepdims=True)) / (mfccs.std(axis=1, keepdims=True) + 1e-10)
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
        features = librosa.power_to_db(mel_spec, ref=np.max)
        features = (features - features.mean()) / features.std()
        features = features.T
    
    elif feature_type == "lpc":
        # Robust LPC feature extraction
        lpc_order = 2
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

def dtw_distance(seq1, seq2):
    """Calculate DTW distance using cosine distance and dynamic programming"""
    dist_mat = dist.cdist(seq1, seq2, "cosine")
    _, cost_mat = dp(dist_mat)
    
    # Normalize alignment cost
    normalized_cost = cost_mat[-1, -1] / (seq1.shape[0] + seq2.shape[0])
    return normalized_cost

def recognize_word(input_file, input_speaker_name, speaker_files, mfcc_threshold = 0.3):
    """
    Recognize word with improved scoring mechanism and pre-filtering
    """
    feature_types = ['mfcc', 'lpc', 'mel']
    feature_results = {}
    
    # Collect unique words in the dataset
    dataset_words = set()
    for words_files in speaker_files.values():
        dataset_words.update(words_files.keys())
    
    # To store distances for calculating averages later
    word_distances_by_feature = {feature: [] for feature in feature_types}
    word_distances_by_speaker = {feature: [] for feature in feature_types}
    
    for feature_type in feature_types:
        input_features = extract_features(input_file, feature_type)
        feature_matches = {}
        
        for speaker, words_files in speaker_files.items():
            if input_speaker_name == speaker:
                continue
            
            speaker_word_distances = []
            
            for word, file_path in words_files.items():
                ref_features = extract_features(file_path, feature_type)
                distance = dtw_distance(input_features, ref_features)

                speaker_word_distances.append((word, distance))
                
                # Collect distances for the predicted word later
                if word in dataset_words:
                    word_distances_by_feature[feature_type].append(distance)
                    word_distances_by_speaker[feature_type].append(distance)
            
            # Normalize distances for pre-filtered matches
            if speaker_word_distances:
                distances = [d[1] for d in speaker_word_distances]
                normalized_distances = 1 - (distances - min(distances)) / (max(distances) - min(distances))
                
                feature_matches[speaker] = [
                    (speaker_word_distances[i][0], normalized_distances[i]) 
                    for i in range(len(speaker_word_distances))
                ]
        
        feature_results[feature_type] = feature_matches
    
    # Aggregate results with voting
    word_scores = {}
    for feature_type, matches in feature_results.items():
        for speaker, word_list in matches.items():
            for word, score in word_list:
                if word not in word_scores:
                    word_scores[word] = []
                word_scores[word].append(score)
    
    # Compute final word ranking
    ranked_words = [
        (word, np.mean(scores)) 
        for word, scores in word_scores.items() 
        if word in dataset_words and len(scores) > 0
    ]
    
    ranked_words.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top-ranked word
    top_word = ranked_words[0][0] if ranked_words else None
    final_top_word = top_word
    
    if top_word:
        print(f"Top predicted word: {top_word}")
        
        # Calculate and print the average DTW distances
        for feature_type in feature_types:
            avg_distance = np.mean(word_distances_by_feature[feature_type])
            # print(f"Average {feature_type.upper()} DTW distance for word '{top_word}': {avg_distance:.4f}")

            if feature_type=='mfcc' and avg_distance > mfcc_threshold:
                final_top_word = None
        
    return final_top_word, top_word



def find_similar_speaker(input_speaker_name, speaker_files):
    """
    Find the most similar speaker by scaling feature distances to ensure equal importance.

    Args:
        input_speaker_name (str): Name of the input speaker.
        speaker_files (dict): Dictionary of speakers with their words and corresponding file paths.

    Returns:
        str: Name of the most similar speaker.
    """
    feature_types = ['mfcc', 'lpc', 'mel']
    speaker_distances = {speaker: {feature: [] for feature in feature_types} for speaker in speaker_files}

    # Collect words from the input speaker
    input_words_files = speaker_files.get(input_speaker_name, {})
    if not input_words_files:
        raise ValueError(f"Speaker '{input_speaker_name}' not found in speaker files.")

    # Extract features for input speaker's words
    input_features = {feature: {} for feature in feature_types}
    for feature_type in feature_types:
        for word, file_path in input_words_files.items():
            input_features[feature_type][word] = extract_features(file_path, feature_type)

    # Compare with other speakers' words
    for speaker, words_files in speaker_files.items():
        if input_speaker_name == speaker:
            continue  # Skip comparing with the input speaker itself

        # Calculate distances for each word and feature type
        for word, file_path in words_files.items():
            if word in input_words_files:  # Only compare corresponding words
                for feature_type in feature_types:
                    # Extract features for the reference word
                    ref_features = extract_features(file_path, feature_type)
                    input_feat = input_features[feature_type][word]

                    # Calculate DTW distance for this feature type
                    distance = dtw_distance(input_feat, ref_features)
                    speaker_distances[speaker][feature_type].append(distance)

    # Normalize distances for each feature type across all speakers
    normalized_distances = {speaker: [] for speaker in speaker_files if speaker != input_speaker_name}
    for feature_type in feature_types:
        # Gather all distances for the current feature type
        all_distances = [
            dist for speaker in speaker_distances if speaker != input_speaker_name
            for dist in speaker_distances[speaker][feature_type]
        ]

        # Normalize using z-score normalization
        mean = np.mean(all_distances)
        std = np.std(all_distances) if np.std(all_distances) > 0 else 1  # Prevent division by zero

        for speaker in speaker_distances:
            if speaker == input_speaker_name:
                continue
            # Normalize each speaker's distances for this feature type
            normalized = [(dist - mean) / std for dist in speaker_distances[speaker][feature_type]]
            normalized_distances[speaker].extend(normalized)

    # Calculate average Euclidean distances for each speaker
    speaker_avg_distances = {}
    for speaker, distances in normalized_distances.items():
        if distances:
            # Compute the average Euclidean distance over all words
            euclidean_distances = [
                np.sqrt(sum(d**2 for d in distances[i:i + 3]))
                for i in range(0, len(distances), 3)  # Group by 3 (one for each feature)
            ]
            speaker_avg_distances[speaker] = np.mean(euclidean_distances)

    # Find the most similar speaker
    most_similar_speaker = min(speaker_avg_distances, key=speaker_avg_distances.get)

    # Print distances and the most similar speaker
    print(f"\nDistance calculations between '{input_speaker_name}' and other speakers:")
    for speaker, avg_distance in speaker_avg_distances.items():
        print(f"  Speaker: {speaker}, Average Euclidean distance: {avg_distance:.4f}")

    print(f"\nMost similar speaker: {most_similar_speaker} with average distance {speaker_avg_distances[most_similar_speaker]:.4f}")
    
    return most_similar_speaker
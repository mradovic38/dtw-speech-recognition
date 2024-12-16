import os
import numpy as np
from typing import Tuple, Optional, Dict, List

from feature_extraction import extract_features
from utils import load_speakers_from_folder
from dtw import calculate_dtw_cost

def recognize_word(input_file: str, sound_database: str, mel_cost_threshold: float = 0.2, confidence_threshold: float = 0.8,
                   input_speaker_name: str = '') -> Tuple[Optional[str], str, Dict[str, List[Tuple[str, float]]]]:
    """
    Recognize a word from an input audio file by comparing it with a sound database.

    Args:
        input_file (str): Path to the input audio file.
        sound_database (str): Path to the folder containing speaker audio files.
        mel_cost_threshold (float): Threshold for Mel spectrogram cost to classify 
                                     a word as unknown. Default is 0.2.
        confidence_threshold (float): Minimum confidence required to classify a word. 
                                       Default is 0.8.
        input_speaker_name (str): Name of the input speaker to exclude from matching. 
                                  Default is an empty string.

    Returns:
        tuple: 
            - final_word (Optional[str]): The recognized word or None if confidence is below 
                                          the threshold or Mel cost exceeds the threshold.
            - predicted_word (str): The final predicted word, regardless of confidence.
            - sorted_dtw_results (dict): Sorted DTW results for each feature type (key)
                                         containing a list of (word, normalized_cost) tuples.
    """

    feature_types = ['mfcc', 'lpc', 'mel']
    weights = {'mfcc': 2, 'mel': 2, 'lpc': 1}

    # Load speaker files and initialize variables
    speaker_files = load_speakers_from_folder(sound_database)
    vocab = np.unique([x.split('-')[0] for x in os.listdir(sound_database)])
    word_costs = {feature: {word: [] for word in vocab} for feature in feature_types}

    # Calculate DTW costs for each feature type
    for feature_type in feature_types:
        input_features = extract_features(input_file, feature_type)

        for speaker in speaker_files:
            if input_speaker_name == speaker:
                continue
            
            for word in speaker_files[speaker]:
                if word not in vocab:
                    continue
                
                file_path = speaker_files[speaker][word]
                ref_features = extract_features(file_path, feature_type)
                
                # Calculate DTW cost
                cost = calculate_dtw_cost(input_features, ref_features)
                
                
                word_costs[feature_type][word].append(cost)

    # Calculate mean normalized costs per word for each feature type
    mean_costs = {feature: {} for feature in feature_types}
    for feature_type in feature_types:
        for word in vocab:
            costs = word_costs[feature_type][word]
            mean_costs[feature_type][word] = np.mean(costs) if costs else np.inf

    # Sort mean costs for each feature type
    sorted_dtw_results = {
        feature: sorted(mean_costs[feature].items(), key=lambda x: x[1])
        for feature in feature_types
    }

    # Generate predictions for each feature type
    predictions = {}
    feature_votes = []
    for feature_type in feature_types:
        best_word = min(mean_costs[feature_type], key=mean_costs[feature_type].get)
        predictions[feature_type] = best_word
        feature_votes.append(feature_type)

    # Weighted voting
    word_scores = {word: 0 for word in vocab}
    for feature_type, predicted_word in predictions.items():
        if predicted_word:
            word_scores[predicted_word] += weights[feature_type]

    # Final word selection based on scores
    final_word = max(word_scores, key=word_scores.get)
    total_score = sum(word_scores.values())
    confidence = word_scores[final_word] / total_score if total_score > 0 else 0

    # Confidence thresholding
    if confidence < confidence_threshold:
        return None, final_word, sorted_dtw_results

    # Check if Mel cost exceeds the threshold
    mel_best_word = predictions['mel']
    if mel_best_word:
        mel_best_cost = mean_costs['mel'][mel_best_word]
        if mel_best_cost > mel_cost_threshold:
            return None, final_word, sorted_dtw_results

    return final_word, final_word, sorted_dtw_results

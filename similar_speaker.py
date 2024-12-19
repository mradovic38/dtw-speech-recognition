import os
import numpy as np

from utils import load_speakers_from_folder
from feature_extraction import extract_features
from dtw import calculate_dtw_cost

def find_similar_speaker(target_speaker: str, sound_database: str, feature_type: str='mfcc') -> str:
    """
    Find the speaker most similar to the target speaker based on DTW costs 
    calculated using extracted features.

    This function compares the target speaker's words to other speakers in the 
    sound database using Dynamic Time Warping (DTW) to compute similarity. The 
    speaker with the smallest average DTW cost is considered the most similar.

    Args:
        target_speaker (str): Name of the target speaker whose similarity is to be evaluated.
        sound_database (str): Path to the folder containing speaker audio files.
        feature_type (str): Type of features to extract for comparison 
                            ('mfcc', 'lpc', 'mel'). Default is 'mfcc'.

    Returns:
        str: The name of the speaker most similar to the target speaker.
    """

    # Load speaker files from the sound database
    speaker_files = load_speakers_from_folder(sound_database)
    
    # Extract features for the target speaker's words
    target_features = {word: extract_features(file_path, feature_type) for word, file_path in speaker_files[target_speaker].items()}

    # Initialize a dictionary to store DTW costs for each speaker
    speaker_dtw_distances = {}
    vocab = np.unique([x.split('-')[0] for x in os.listdir(sound_database)])

    # Loop through all speakers in the dataset
    for speaker, words in speaker_files.items():
        if speaker == target_speaker:
            continue
        
        # Initialize a list to accumulate the DTW distances for each word
        total_cost = 0
        
        for word, file_path in words.items():
            if word in target_features:
                # Extract features for the reference word
                ref_features = extract_features(file_path, feature_type)
                
                # Calculate the DTW cost between the target word and the reference word
                cost = calculate_dtw_cost(target_features[word], ref_features)
                
                total_cost += cost
        
        # Compute the average DTW cost for this speaker
        average_cost = total_cost / len(vocab)
        speaker_dtw_distances[speaker] = average_cost

    # Find the speaker with the minimum DTW distance
    most_similar_speaker = min(speaker_dtw_distances, key=speaker_dtw_distances.get)
    most_similar_distance = speaker_dtw_distances[most_similar_speaker]
    
    return most_similar_speaker , most_similar_distance
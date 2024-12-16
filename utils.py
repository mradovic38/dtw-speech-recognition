import os
from typing import Dict, List, Tuple


def load_speakers_from_folder(folder_path: str) -> Dict[str, Dict[str, str]]:
    """
    Reads all audio files in a folder and creates a dictionary of speakers.
    The outer dictionary has speaker names as keys.
    The inner dictionary has words as keys and file paths as values.

    Args:
        folder_path (str): Path to the folder containing audio files.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary of speakers and their recordings.
    """
    speakers = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            # Extract word and speaker name from the file name
            base_name = file_name.replace(".wav", "")
            word, speaker_name = base_name.split("-", 1)

            # Add speaker to the dictionary if not already present
            if speaker_name not in speakers:
                speakers[speaker_name] = {}

            # Add the file path to the inner dictionary for the corresponding word
            speakers[speaker_name][word] = file_path

    return speakers


def print_results(recognized_word: str, top_pred: str, sorted_dtw_results: Dict[str, List[Tuple[str, float]]]) -> None:
    '''
    Prints results in a prettier way.

    Args:
        recognized word (str): Word that was detected.
        top_pred (str): Top prediction word.
    '''
    print("Sorted DTW Results:")
    for ft, res_list in sorted_dtw_results.items():
        print(f"\t{ft.upper()}:")
        for word, cost in res_list:
            print(f"\t\tWord: {word}, Cost: {cost:.4f}")
            
    if recognized_word:
        print(f"Word detected: {recognized_word}")

    else:
        print(f"Word not detected! Top prediction: {top_pred}")

    



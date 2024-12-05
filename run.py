from dtw import recognize_word, find_similar_speaker
from file_utils import load_speakers_from_folder

DB_PATH = "sound_database"


speakers = load_speakers_from_folder(DB_PATH)


print("Results on a word from a speaker who's sounds are in the dataset, but the said word is not in the dataset vocabulary:")
REF_NAME = ""
INPUT_FILE = "test.wav"
recognized_word = recognize_word(INPUT_FILE, REF_NAME, speakers)
print(f"Recognized word: {recognized_word}")


print("\nResults on a word from a speaker who's sounds are not in the dataset, but the said word is in the dataset vocabulary:")
REF_NAME = ""
INPUT_FILE = 'desno-test.wav'
recognized_word = recognize_word(INPUT_FILE, REF_NAME, speakers)
print(f"Recognized word: {recognized_word}")


print("\nResults on a word from a speaker who's sounds are in the dataset and the said word is in the dataset vocabulary:")
REF_NAME = ""
INPUT_FILE = 'levo-test.wav'
recognized_word = recognize_word(INPUT_FILE, REF_NAME, speakers)
print(f"Recognized word: {recognized_word}")
from word_recognition import recognize_word
from similar_speaker import find_similar_speaker
from utils import print_results


if __name__ == '__main__':

    DB_PATH = "sound_database"
    REF_NAME = '38-21'
    

    print("Results on a word from a speaker who's sounds are in the dataset, but the said word is not in the dataset vocabulary:")
    INPUT_FILE = "test.wav"
    recognized_word, top_word = recognize_word(INPUT_FILE, DB_PATH)
    print_results(recognized_word, top_word)

    print("\nResults on a word from a speaker who's sounds are not in the dataset, but the said word is in the dataset vocabulary:")
    INPUT_FILE = 'desno-test.wav'
    recognized_word, top_word = recognize_word(INPUT_FILE, DB_PATH)
    print_results(recognized_word, top_word)


    print("\nResults on a word from a speaker who's sounds are in the dataset and the said word is in the dataset vocabulary:")
    INPUT_FILE = 'levo-test.wav'
    recognized_word, top_word = recognize_word(INPUT_FILE, DB_PATH)
    print_results(recognized_word, top_word)

    print('\nSimilar speaker to speaker named 38-21 based on MFCC:')

    print(find_similar_speaker(REF_NAME, DB_PATH, feature_type='mfcc'))

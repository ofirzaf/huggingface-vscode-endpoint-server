"""
Develop a Python program that reads all the text files under a directory and 
reprompt top-5 words with the most number of occurrences.
"""

import os
import re
from collections import Counter

def read_files(directory):
    """
    Reads all text files in the given directory and returns a list of their contents.
    """
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def clean_text(text):
    """
    Cleans the text by removing punctuation and converting to lowercase.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_top_words(texts, n=5):
    """
    Combines all texts into one string, cleans it, and finds the top-n most common words.
    """
    combined_text = ' '.join(texts)
    cleaned_text = clean_text(combined_text)
    words = cleaned_text.split()
    word_counts = Counter(words)
    return word_counts.most_common(n)

def main():
    directory = input("Enter the directory path containing text files: ")
    texts = read_files(directory)
    top_words = get_top_words(texts)
    print("Top-5 most common words:")
    for word, count in top_words:
        print(f"{word}: {count}")

if __name__ == "__main__":
    main()






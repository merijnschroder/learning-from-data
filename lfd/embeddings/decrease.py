import json
from collections import Counter

def main():
    embeddings_file = 'glove.twitter.27B.100d.json'
    file_loc = '../../data/original/'
    file_list = [
        f'{file_loc}train.tsv',
        f'{file_loc}dev.tsv',
        f'{file_loc}test.tsv'
    ]
    new_json = {}

    embeddings = json.load(open(embeddings_file, 'r', encoding='utf-8'))

    main_list = []

    for file_name in file_list:
        current_list = []
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                current_list += line.split('\t')[0].split()

        main_list += current_list

    relevant_words = list(Counter(main_list))

    for key, item in embeddings.items():
        if key in relevant_words:
            new_json[key] = item

    file_name = embeddings_file.replace('27B', 'filtered')
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(new_json, file)


if __name__ == '__main__':
    main()

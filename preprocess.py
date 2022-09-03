import re
from collections import defaultdict
from typing import List
import string

file_names = ['reut2-002', 'reut2-009', 'reut2-014', 'reut2-004', 'reut2-013', 'reut2-006', 'reut2-015', 'reut2-011',
              'reut2-005', 'reut2-000', 'reut2-007', 'reut2-001', 'reut2-012', 'reut2-010', 'reut2-008', 'reut2-003',
              'reut2-016', 'reut2-017', 'reut2-018', 'reut2-019', 'reut2-020', 'reut2-021']
folder_name = 'reuters21578'
stopwords_file_name = 'stopwords.txt'


# applies case-fold to input string, removes punctuation and returns the string as tokens
def normalize(text: str) -> [str]:
    text = text.casefold()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [word for word in text.split()]


# removes the stopwords and numbers from the list
def tokenize(tokens: [str], stopwords: [str]) -> [str]:
    return [word for word in tokens if word not in stopwords and not word.isnumeric()]


# reads the stopwords from the file
def read_stopwords() -> List[str]:
    with open(stopwords_file_name) as reader:
        stopwords = [line.lower().replace('\n', '') for line in reader.readlines()]
        reader.close()
    return stopwords


# preprocesses the given data set, reads the files, extracts the tests, normalizes and stores into a dictionary
def preprocess():
    stopwords = read_stopwords()
    vocab = set()  # distinct tokens in entire dataset
    classes = defaultdict(list)  # {class_name : [new_ids]} for train dataset
    test_classes, train_classes = dict(), dict()  # {new_id = [assigned classes]}
    train, test = {}, {}  # {new_id = [normalized tokens]}

    for file_name in file_names:
        # traverse the sgm files one by one, read the contents
        with open(f'{folder_name}/{file_name}.sgm', encoding="latin-1") as reader:
            content = reader.read().replace('\n', ' ')
            articles = re.findall("<TEXT(.*?)</TEXT>", content)  # find the texts of all news stories
            new_ids = re.findall("NEWID=\"(\d+)\"", content)  # find the new_ids for all articles
            train_or_tests = re.findall("LEWISSPLIT=\"(.*?)\"", content)
            topics_list = re.findall("<TOPICS>(.*?)</TOPICS>", content)

            # for each article, extract the title and body fields, normalize, tokenize and merge the texts
            for i, article in enumerate(articles):
                title_reg = re.findall("<TITLE(.*?)</TITLE>", article)
                title = title_reg[0] if len(title_reg) > 0 else ' '
                body_reg = re.findall("<BODY(.*?)</BODY>", article)
                body = body_reg[0] if len(body_reg) > 0 else ' '
                train_or_test = train_or_tests[i]
                topics = re.findall("<D>(.*?)</D>", topics_list[i])
                normalized = [*tokenize(normalize(title), stopwords), *tokenize(normalize(body), stopwords)]
                vocab.update(normalized)  # add tokens to vocabulary
                new_id = int(new_ids[i])
                # build the test and train dictionaries based on the LEWISSPLIT value
                if train_or_test == 'TRAIN':
                    for topic in topics:
                        classes[topic].append(new_id)
                    train[new_id] = normalized
                    train_classes[new_id] = topics
                elif train_or_test == 'TEST':
                    test[new_id] = normalized
                    test_classes[new_id] = topics

    # identify the most common 10 topics in the corpus -using train dataset only-
    # select as a dataset the articles that belong to one or more of these 10 topics
    classes = dict(classes)
    sorted_classes = sorted(classes.items(), key=lambda x: len(x[1]), reverse=True)
    most_common_classes = sorted_classes[:10] # {class_name : [new_ids]} for train dataset
    selected_classes = [c[0] for c in most_common_classes]

    # remove the documents that do not belong to any one of the top 10 classes from the dataset
    cleaned_vocab = set()
    train_cleaned = {}
    for new_id, normalized in train.items():
        classes = train_classes[new_id]
        if not len(set(classes).intersection(set(selected_classes))) == 0:
            train_cleaned[new_id] = normalized
            cleaned_vocab.update(normalized)
    test_cleaned = {}
    for new_id, normalized in test.items():
        classes = test_classes[new_id]
        if not len(set(classes).intersection(set(selected_classes))) == 0:
            test_cleaned[new_id] = normalized
            cleaned_vocab.update(normalized)

    # print information about the dataset

    # size of vocabulary
    print(f'Size of vocabulary after preprocessing = {len(vocab)}')
    print(f'Size of vocabulary in selected documents = {len(cleaned_vocab)}')

   # size of train and test datasets
    print(f'\nTrain Dataset Size = {len(train_cleaned)}')
    print(f'Test Dataset Size = {len(test_cleaned)}')

    # number of train and test documents in each selected class
    print('\nSELECTED CLASSES')
    for cls in selected_classes:
        train_count, test_count = 0, 0
        for new_id in train_cleaned:
            classes = train_classes[new_id]
            if cls in classes:
                train_count += 1
        for new_id in test_cleaned:
            classes = test_classes[new_id]
            if cls in classes:
                test_count += 1
        print('{:>10} '.format(cls), f'train={train_count}\ttest={test_count}')

    # number of train and test documents that belong to more than one selected class
    labelled_more_train, labelled_more_test = 0, 0
    for new_id in train_cleaned:
        classes = train_classes[new_id]
        if len(set(classes).intersection(set(selected_classes)))>1:
            labelled_more_train +=1
    for new_id in test_cleaned:
        classes = test_classes[new_id]
        if len(set(classes).intersection(set(selected_classes)))>1:
            labelled_more_test +=1
    print(f'Train Documents in Multiple Classes  = {labelled_more_train}')
    print(f'Test Documents in Multiple Classes  = {labelled_more_test}')
    return train_cleaned, test_cleaned, most_common_classes, train_classes, test_classes


if __name__ == '__main__':
    preprocess()

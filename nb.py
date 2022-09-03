import csv
import math
from collections import defaultdict
from preprocess import preprocess


def nb_learner(train, most_common_classes):
    vocabulary = defaultdict(int)  # bag of words for train dataset
    mega_documents = dict()  # {class : {token: count}} for train dataset
    prior_probabilities = defaultdict(float)  # {class: prior probability}
    len_train_documents = 0
    for cls, train_documents in most_common_classes:  # [(class_name, [new_ids])] for train dataset
        prior_probabilities[cls] = len(train_documents)  # number of train documents for that class
        len_train_documents += len(train_documents)

        # form a mega document from each document belonging to that label
        mega_document = defaultdict(int)
        for doc in train_documents:
            document_words = train[doc]
            for word in document_words:
                mega_document[word] += 1
                vocabulary[word] += 1
        mega_documents[cls] = mega_document
    return mega_documents, prior_probabilities, len_train_documents, vocabulary


def nb_classifier(mega_documents, prior_probabilities, len_train_documents, vocabulary, test):
    results = {}
    possible_classes = list(mega_documents.keys())

    for new_id, words in test.items():
        res = {}
        for possible_class in possible_classes:
            probability = math.log(prior_probabilities[possible_class] / len_train_documents)
            total_words_for_class = sum(frq for frq in mega_documents[possible_class].values())
            denominator = total_words_for_class + len(vocabulary)
            for word in words:
                nominator = mega_documents[possible_class].get(word, 0) + 1
                likelihood = nominator / denominator
                probability += math.log(likelihood)
            res[possible_class] = probability

        results[new_id] = res

    return results, possible_classes


def nb_print_outputs(results, test_classes, possible_classes, range):
    # open two csv files to write result and real classes for test documents
    result_file = open('nb_result_targets.csv', 'w')
    result_writer = csv.writer(result_file)
    real_file = open('nb_real_targets.csv', 'w')
    real_writer = csv.writer(real_file)
    first_row = ['new_id'] + possible_classes
    result_writer.writerow(first_row)
    real_writer.writerow(first_row)

    for new_id, probabilities in results.items():
        # to the result file, write new id, write 1 for spotted classes, 0 for others
        # assign the document to classes within the 2.5% highest probability range
        row = [new_id]
        max_val = max(probabilities.values())
        for cls, y in probabilities.items():
            if -100 * (max_val - y) / max_val < range:
                row.append(1)
            else:
                row.append(0)
        result_writer.writerow(row)

        # to the real file, write new id, write 1 for really included classes, 0 for others
        # read the real classes from the preprocess module output
        real_row = [new_id]
        real_y = test_classes[int(new_id)]
        for cls in possible_classes:
            if cls in real_y:
                real_row.append(1)
            else:
                real_row.append(0)
        real_writer.writerow(real_row)

    result_file.close()
    real_file.close()

    f = open("classes.txt", "w")
    for cls in possible_classes:
        f.write(cls)
        f.write('\n')
    f.close()


# multinomial NB classifier with add 1 smoothing
def nb():
    train, test, most_common_classes, train_classes, test_classes = preprocess()
    mega_documents, prior_probabilities, len_train_documents, vocabulary = nb_learner(train, most_common_classes)
    results, possible_classes = nb_classifier(mega_documents, prior_probabilities, len_train_documents, vocabulary,
                                              test)
    nb_print_outputs(results, test_classes, possible_classes, 2.5)


if __name__ == "__main__":
    nb()

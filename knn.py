import csv
import math
from collections import defaultdict
from preprocess import preprocess


# calculates the length of a vector
def calculate_vector_length(vector: dict) -> float:
    length = 0
    for k, v in vector.items():
        length += v ** 2
    return math.sqrt(length)


def build_frequency_index(index: dict):
    index_with_frequencies = {}
    for token, value in index.items():
        documents = {}
        for new_id, pos in value.items():
            documents[new_id] = {'tf': len(pos), 'positions': list(pos)}
        new_value = {'df': len(value), 'documents': documents}
        index_with_frequencies[token] = new_value
    return index_with_frequencies


def knn_learner(train):
    inverted_index = defaultdict(lambda: defaultdict(list))  # {token: {new_id_1: [pos_1, pos_2]}, ..}
    document_index = defaultdict(lambda: defaultdict(int))  # { new_id: {token: document_frequency }}
    # build frequency indexes
    for new_id, tokens in train.items():
        for i, token in enumerate(tokens):
            inverted_index[token][new_id].append(i)
            document_index[new_id][token] += 1
    index_with_frequencies = build_frequency_index(inverted_index)

    idf_vector = {}  # {token: (log-scaled)IDF vector}
    N = len(train)
    for token, docs in index_with_frequencies.items():
        df = docs.get('df')
        idf_weight = math.log((N + 1) / df, 10) if (df is not None and not df == 0) else 0.0
        idf_vector[token] = idf_weight

    # tf-idf based cosine similarity
    # calculate idf-tf vector for each document
    document_idf_tfs = {}  # {new_id: {token: (ls)IDF-TF }}
    for new_id, document in document_index.items():
        document_idf_tf = {}
        for token, tf in document.items():
            document_tf = 1 + math.log(tf, 10) if (tf is not None and not tf == 0) else 0.0
            idf_value = idf_vector.get(token)
            idf_value = idf_value if idf_value is not None else 0.0
            document_idf_tf[token] = document_tf * idf_value
        document_vector_length = calculate_vector_length(document_idf_tf)
        document_idf_tfs[int(new_id)] = {'length': document_vector_length, 'vectors': document_idf_tf}
    return idf_vector, document_idf_tfs


def knn_classifier(idf_vector, document_idf_tfs, most_common_classes, train_classes, test, k):
    selected_classes = [c[0] for c in most_common_classes]

    test_tf_idfs = {}
    for new_id, doc in test.items():
        test_idf_tf = {}  # {token: (ls)IDF-TF }
        for token in doc:
            test_tf = doc.count(token)
            test_tf_weight = 1.0 if test_tf == 0 else 1 + math.log(test_tf, 10)
            idf_value = idf_vector.get(token)
            idf_value = idf_value if idf_value is not None else 0.0
            test_idf_tf[token] = test_tf_weight * idf_value
        test_tf_idfs[new_id] = test_idf_tf

    results = {}  # {new_id: {class: neighbor count in k nearest in that class}}
    # calculate cosine similarity of each document idf-tf vector with the query idf-tf vector
    for test_new_id, test_idf_tf in test_tf_idfs.items():
        document_scores = {}
        for new_id, document_idf_tf in document_idf_tfs.items():
            cos_value = 0
            for token, doc_val in document_idf_tf['vectors'].items():
                test_val = test_idf_tf.get(token)
                test_val = test_val if test_val is not None else 0.0
                cos_value += doc_val * test_val
            if cos_value > 0:
                document_scores[new_id] = cos_value / (document_idf_tf['length'] * calculate_vector_length(test_idf_tf))

        # after calculating the cos similarity of the test document with train documents, sort them according to score
        # take the k neartest neighbors
        sorted_dict = dict(sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:k])
        classes = defaultdict(int)
        for new_id, cosine_score in sorted_dict.items():
            for c in train_classes[new_id]:
                if c in selected_classes:
                    classes[c] += 1
            results[test_new_id] = classes

    return results, selected_classes


def knn_print_outputs(results, test_classes, possible_classes, range):
    # open two csv files to write result and real classes for test documents
    result_file = open('knn_result_targets.csv', 'w')
    result_writer = csv.writer(result_file)
    real_file = open('knn_real_targets.csv', 'w')
    real_writer = csv.writer(real_file)
    first_row = ['new_id'] + possible_classes
    result_writer.writerow(first_row)
    real_writer.writerow(first_row)

    for new_id, probabilities in results.items():
        # assign the document to classes within the 30% highest included neighbor
        row_result = set()
        max_val = max(probabilities.values())
        for cls, n in probabilities.items():
            if 100 * (max_val - n) / max_val <= range:
                row_result.add(cls)

        # to the result file, write new id, write 1 for spotted classes, 0 for others
        # to the real file, write new id, write 1 for really included classes, 0 for others
        # read the real classes from the preprocess module output
        row = [new_id]
        real_row = [new_id]
        real_y = test_classes[int(new_id)]
        for cls in possible_classes:
            if cls in real_y:
                real_row.append(1)
            else:
                real_row.append(0)
            if cls in row_result:
                row.append(1)
            else:
                row.append(0)
        result_writer.writerow(row)
        real_writer.writerow(real_row)

    result_file.close()
    real_file.close()

    f = open("classes.txt", "w")
    for cls in possible_classes:
        f.write(cls)
        f.write('\n')
    f.close()


def knn():
    train, test, most_common_classes, train_classes, test_classes = preprocess()
    idf_vector, document_idf_tfs = knn_learner(train)
    results, selected_classes = knn_classifier(idf_vector, document_idf_tfs, most_common_classes, train_classes, test, 7)
    knn_print_outputs(results, test_classes, selected_classes, 40)


if __name__ == "__main__":
    knn()

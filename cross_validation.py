from evaluate_results import evaluate
from knn import knn_print_outputs, knn_learner, knn_classifier
from nb import nb_classifier, nb_learner, nb_print_outputs
from preprocess import preprocess


def try_knn(k, range, train, test, most_common_classes, train_classes, test_classes):
    idf_vector, document_idf_tfs = knn_learner(train)
    results, selected_classes = knn_classifier(idf_vector, document_idf_tfs, most_common_classes, train_classes, test,
                                               k)
    knn_print_outputs(results, test_classes, selected_classes, range)
    evaluate('knn')
    with open('f_scores_knn.txt') as f:
        lines = f.readlines()
        macro = float(lines[0].strip())
        micro = float(lines[1].strip())
    f.close()
    print(f'k={k} range={range} macro={macro}, micro={micro}')
    return macro, micro


def try_nb(range, train, test, most_common_classes, test_classes):
    mega_documents, prior_probabilities, len_train_documents, vocabulary = nb_learner(train, most_common_classes)
    results, selected_classes = nb_classifier(mega_documents, prior_probabilities, len_train_documents, vocabulary,
                                              test)
    nb_print_outputs(results, test_classes, selected_classes, range)
    evaluate('nb')
    with open('f_scores_nb.txt') as f:
        lines = f.readlines()
        macro = float(lines[0].strip())
        micro = float(lines[1].strip())
    f.close()
    print(f'range={range} macro={macro}, micro={micro}')
    return macro, micro


def cross_validate_knn(train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes):
    # try the knn algorithm with different k values using the train and dev datasets
    res = ['F scores for knn']
    macro, micro = try_knn(1, 30, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={1} range={30} macro={macro}, micro={micro}')
    macro, micro = try_knn(3, 30, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={3} range={30} macro={macro}, micro={micro}')
    macro, micro = try_knn(5, 30, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={5} range={30} macro={macro}, micro={micro}')
    macro, micro = try_knn(7, 30, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={7} range={30} macro={macro}, micro={micro}')
    macro, micro = try_knn(9, 30, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={9} range={30} macro={macro}, micro={micro}')

    macro, micro = try_knn(1, 35, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={1} range={35} macro={macro}, micro={micro}')
    macro, micro = try_knn(3, 35, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={3} range={35} macro={macro}, micro={micro}')
    macro, micro = try_knn(5, 35, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={5} range={35} macro={macro}, micro={micro}')
    macro, micro = try_knn(7, 35, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={7} range={35} macro={macro}, micro={micro}')
    macro, micro = try_knn(9, 35, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={9} range={35} macro={macro}, micro={micro}')

    macro, micro = try_knn(1, 40, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={1} range={40} macro={macro}, micro={micro}')
    macro, micro = try_knn(3, 40, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={3} range={40} macro={macro}, micro={micro}')
    macro, micro = try_knn(5, 40, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={5} range={40} macro={macro}, micro={micro}')
    macro, micro = try_knn(7, 40, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={7} range={40} macro={macro}, micro={micro}')
    macro, micro = try_knn(9, 40, train_dataset, dev_dataset, classes, train_dataset_classes, dev_dataset_classes)
    res.append(f'k={9} range={40} macro={macro}, micro={micro}')


    f = open("cross_validation_knn.txt", "w")
    for res_line in res:
        f.write(res_line)
        f.write('\n')
    f.close()


def cross_validate_nb(train_dataset, dev_dataset, classes, dev_dataset_classes):

    res = ['F scores for NB']
    macro, micro = try_nb(1, train_dataset, dev_dataset, classes, dev_dataset_classes)
    res.append(f'range={1} macro={macro}, micro={micro}')
    macro, micro = try_nb(1.5, train_dataset, dev_dataset, classes, dev_dataset_classes)
    res.append(f'range={1.5} macro={macro}, micro={micro}')
    macro, micro = try_nb(2, train_dataset, dev_dataset, classes, dev_dataset_classes)
    res.append(f'range={2} macro={macro}, micro={micro}')
    macro, micro = try_nb(2.5, train_dataset, dev_dataset, classes, dev_dataset_classes)
    res.append(f'range={2.5} macro={macro}, micro={micro}')
    macro, micro = try_nb(3, train_dataset, dev_dataset, classes, dev_dataset_classes)
    res.append(f'range={3} macro={macro}, micro={micro}')
    macro, micro = try_nb(3.5, train_dataset, dev_dataset, classes, dev_dataset_classes)
    res.append(f'range={3.5} macro={macro}, micro={micro}')
    macro, micro = try_nb(4, train_dataset, dev_dataset, classes, dev_dataset_classes)
    res.append(f'range={4} macro={macro}, micro={micro}')
    macro, micro = try_nb(4.5, train_dataset, dev_dataset, classes, dev_dataset_classes)
    res.append(f'range={4.5} macro={macro}, micro={micro}')
    macro, micro = try_nb(5, train_dataset, dev_dataset, classes, dev_dataset_classes)
    res.append(f'range={5} macro={macro}, micro={micro}')

    f = open("cross_validation_nb.txt", "w")
    for res_line in res:
        f.write(res_line)
        f.write('\n')
    f.close()


def cross_validate():
    train, test, most_common_classes, train_classes, test_classes = preprocess()

    # separate part of the training set as dev set
    dev_dataset_size = int((len(list(train.keys())) / 10))
    dev_dataset_new_ids = list(train.keys())[:dev_dataset_size]
    train_dataset_new_ids = list(train.keys())[dev_dataset_size:]

    # rebuild the train and dev dataset and classes
    train_dataset, dev_dataset = {}, {}
    for new_id, normalized in train.items():
        if new_id in train_dataset_new_ids:
            train_dataset[new_id] = normalized
        elif new_id in dev_dataset_new_ids:
            dev_dataset[new_id] = normalized

    train_dataset_classes, dev_dataset_classes = {}, {}
    for new_id, classes in train_classes.items():
        if new_id in train_dataset_new_ids:
            train_dataset_classes[new_id] = classes
        elif new_id in dev_dataset_new_ids:
            dev_dataset_classes[new_id] = classes

    most_common_classes_updated = []
    for c, docs in most_common_classes:
        docs_updated = []
        for doc in docs:
            if doc in train_dataset_new_ids:
                docs_updated.append(doc)
        most_common_classes_updated.append((c,docs_updated))

    cross_validate_knn(train_dataset, dev_dataset, most_common_classes, train_dataset_classes, dev_dataset_classes)
    cross_validate_nb(train_dataset, dev_dataset, most_common_classes_updated, dev_dataset_classes)


if __name__ == "__main__":
    cross_validate()

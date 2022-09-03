import csv
import sys


def calculate_precision(confusion_matrix):
    precisions = {}
    nom, denom = 0, 0
    for cls, conf in confusion_matrix.items():
        nom += conf['TP']
        denom += (conf['TP'] + conf['FP'])
        precision = conf['TP'] / (conf['TP'] + conf['FP'])
        precisions[cls] = precision
    macro_average = sum(precisions.values()) / len(precisions)
    micro_average = nom / denom
    return precisions, macro_average, micro_average


def calculate_accuracy(confusion_matrix):
    accuracies = {}
    nom, denom = 0, 0
    for cls, conf in confusion_matrix.items():
        nom += (conf['TP'] + conf['TN'])
        denom += sum(conf.values())
        accuracy = (conf['TP'] + conf['TN']) / sum(conf.values())
        accuracies[cls] = accuracy
    macro_average = sum(accuracies.values()) / len(accuracies)
    micro_average = nom / denom
    return accuracies, macro_average, micro_average


def calculate_recall(confusion_matrix):
    recalls = {}
    nom, denom = 0, 0
    for cls, conf in confusion_matrix.items():
        nom += conf['TP']
        denom += (conf['TP'] + conf['FN'])
        recall = conf['TP'] / (conf['TP'] + conf['FN'])
        recalls[cls] = recall
    macro_average = sum(recalls.values()) / len(recalls)
    micro_average = nom / denom
    return recalls, macro_average, micro_average


def calculate_f1(confusion_matrix):
    f1s = {}
    nom, denom = 0, 0
    recalls, macro_average_recalls, micro_average_recalls = calculate_recall(confusion_matrix)
    precisions, macro_average_precisions, micro_average_precisions = calculate_precision(confusion_matrix)
    for cls, conf in confusion_matrix.items():
        nom += (2 * recalls[cls] * precisions[cls])
        denom += (recalls[cls] + precisions[cls])
        f1 = (2 * recalls[cls] * precisions[cls]) / (recalls[cls] + precisions[cls])
        f1s[cls] = f1
    macro_average = sum(f1s.values()) / len(f1s)
    micro_average = nom / denom
    return f1s, macro_average, micro_average


def print_scores(name, scores, macro, micro):
    print('\n', name)
    for cls, score in scores.items():
        print('{:>10} '.format(cls), scores[cls])
    print('{:>10} '.format('macro avg'), macro)
    print('{:>10} '.format('micro avg'), micro)


def evaluate(algorithm, randomized=False, output=True):
    with open('classes.txt') as f:
        lines = f.readlines()
    classes = [line.strip() for line in lines]
    confusion_matrix = {cls: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for cls in classes}

    real_file_name = f'{algorithm}_real_targets.csv'
    result_file_name = f'{algorithm}_result_targets.csv'

    if randomized:
        result_file_name = f'{algorithm}_result_targets_randomized.csv'

    with open(real_file_name) as real_file:
        real_lines = csv.DictReader(real_file)
        with open(result_file_name) as result_file:
            result_lines = csv.DictReader(result_file)

            for real_line, result_line in zip(real_lines, result_lines):
                for cls in classes:
                    real_assigned = real_line[cls] == "1"
                    result_assigned = result_line[cls] == "1"
                    if real_assigned and result_assigned:
                        confusion_matrix[cls]['TP'] += 1
                    elif real_assigned and not result_assigned:
                        confusion_matrix[cls]['FN'] += 1
                    elif not real_assigned and result_assigned:
                        confusion_matrix[cls]['FP'] += 1
                    elif not real_assigned and not result_assigned:
                        confusion_matrix[cls]['TN'] += 1
            if output:
                print('CONFUSION MATRIX')
                for cls, conf in confusion_matrix.items():
                    print(f'\n{cls}')
                    print(f"{conf['TP']}  {conf['FP']}\n{conf['FN']}  {conf['TN']}")

    scores, macro, micro = calculate_precision(confusion_matrix)
    if output:
        print_scores('PRECISION', scores, macro, micro)
    scores, macro, micro = calculate_accuracy(confusion_matrix)
    if output:
        print_scores('ACCURACY', scores, macro, micro)
    scores, macro, micro = calculate_recall(confusion_matrix)
    if output:
        print_scores('RECALL', scores, macro, micro)
    scores, macro, micro = calculate_f1(confusion_matrix)
    if output:
        print_scores('F1', scores, macro, micro)

    return macro


if __name__ == "__main__":
    evaluate(sys.argv[1], False)

import csv
from random import random
from evaluate_results import evaluate

if __name__ == "__main__":
    with open('classes.txt') as f:
        lines = f.readlines()
    classes = [line.strip() for line in lines]

    initial_knn_macro_f1 = evaluate('knn', False, False)
    initial_nb_macro_f1 = evaluate('nb', False, False)

    counter = 0
    for i in range(0,1000):
        f1 = open('knn_result_targets_randomized.csv', 'w')
        knn_writer = csv.writer(f1)
        f2 = open('nb_result_targets_randomized.csv', 'w')
        nb_writer = csv.writer(f2)

        first_row = ['new_id'] + classes
        knn_writer.writerow(first_row)
        nb_writer.writerow(first_row)


        with open('knn_result_targets.csv') as knn_result_file:
            knn_result_lines = csv.DictReader(knn_result_file)
            with open('nb_result_targets.csv') as nb_result_file:
                nb_result_lines = csv.DictReader(nb_result_file)

                for knn_result_line, nb_result_line in zip(knn_result_lines, nb_result_lines):
                    new_id = knn_result_line['new_id']
                    knn_line, nb_line = [new_id], [new_id]
                    for cls in classes:
                        r = random()
                        knn_result = knn_result_line[cls]
                        nb_result = nb_result_line[cls]
                        if r>0.5:
                            knn_line.append(nb_result)
                            nb_line.append(knn_result)
                        else:
                            knn_line.append(knn_result)
                            nb_line.append(nb_result)
                    knn_writer.writerow(knn_line)
                    nb_writer.writerow(nb_line)
        knn_result_file.close()
        nb_result_file.close()
        f1.close()
        f2.close()

        randomized_knn_macro_f1 = evaluate('knn', True, False)
        randomized_nb_macro_f1 = evaluate('nb', True, False)
        if abs(randomized_knn_macro_f1 - randomized_nb_macro_f1) > abs(initial_knn_macro_f1 - initial_nb_macro_f1):
            counter+=1

    print((counter+1)/10001)

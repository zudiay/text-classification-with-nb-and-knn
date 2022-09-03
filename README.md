# Text Classification with Multinomial NB and kNN

Implementation of a multinomial Naive Bayes (NB) and k Nearest Neighbor (kNN) algorithms for text classification.

The Reuters-21578 data set is used.
Reuters-21578 contains 21578 news stories from
Reuters newswire. There are 22 SGML files, each containing 1000 news articles, except the last
file, which contains 578 articles. 
. There are a total of 118 topics (classes) and each article is classified into one
or more topics

The following steps are performed:
1) Pre-processing the Data Set: The text of a news story is enclosed under the <TEXT> tag.
The <TITLE> and the <BODY> fields are used to extract the text of a news story.
A tokenizer is implemented to get the tokens from the news texts and normalization operations are performed including case-folding, stopword removal, and punctuation removal.
2) Creating the training and test sets: 
The top 10 classes (topics) are used in this project. Firstly, the most common 10 topics in the corpus are identified. Then, the articles that belong to one or more of these 10 topics are selected as dataset . The news articles that
are denoted with the LEWISSPLIT=“TRAIN” tag are be inlcuded in the training set and
the articles denoted with the LEWISSPLIT=“TEST” tag are be included in the test set.
3) Implementation of a multinomial NB classifier: The vocabulary and the parameters of the
classifier are learned from the training set. Add-one smoothing is used.
4) Implementation of a kNN classifier: The vocabulary is obtained from the training set and tf-idf
based cosine similarity is used as a similarity function. The best value for k is determined by experimenting with different values of k on the development set. The micro and macro averaged F-score for each tested value of k on the development set is reported.
5) Evaluation: Rhe macro and micro-averaged precision, recall, and F-score values of
the NB and kNN (for the best k value determined in the previous step) algorithms on the test
set are reported.
6) Statistical significance: Randomization test is performed to measure the significance of the difference between the macro-averaged F-scores of the NB and kNN classifiers on the test set.

### Running the program

Python version: Python 3.10.0

Put the reuters21578 folder under the src folder.

Open the terminal in the src folder.

Run the following command to run the NB classifier. It outputs the real and predicted targets to nb_real_targets.csv and nb_result_targets.csv files.

```  python3 nb.py ```

Run the following command to see the results of the NB classifier. It prints the confusion matrix, precision, accuracy, recall and f1 scores to the console.

```  python3 evaluate_results.py nb```

Run the following command to run the KNN classifier.  It outputs the real and predicted targets to knn_real_targets.csv and knn_result_targets.csv files.

```  python3 knn.py ```

Run the following command to see the results of the KNN classifier. It prints the confusion matrix, precision, accuracy, recall and f1 scores to the console.

```  python3 evaluate_results.py knn```

Run the following command to run the preprocess module and see summary data.
```  python3 preprocess.py ```

Run the following command to run the cross validation module. It prints the macro and micro averaged F1 scores for different parameters to cross_validation_knn.txt and cross_validation_nb.txt files.
```  python3 cross_validation.py ```

Run the following command to run the randomization test module. You need to call the knn and nb module before calling this module.
```  python3 randomization_test.py ```


<i> Developed for CMPE493 Introduction to Information Retrieval course, Bogazici University, Fall 2021 <i>

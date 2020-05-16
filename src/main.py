import os
import json
from random import shuffle
from src.pre_processing import Preprocessing
import operator
from src.serialization import Serialization


def dataSplit(jsonlist):
    docs_len = len(jsonlist)    # 737
    train_len = round(70/100 * docs_len)    # 516
    test_len = docs_len - train_len     # 221
    trainset = []
    testset = []

    for f in range(0, train_len):
        trainset.append(jsonlist[f].copy())

    for f in range(train_len, docs_len):
        testset.append(jsonlist[f].copy())

    jsonlist.clear()

    return trainset, testset


def createVocabulary(trainset):
    vocab = []
    for file in trainset:
        for feature in file["features"].keys():
            if feature not in vocab:
                vocab.append(feature)
    return vocab


if __name__ == "__main__":

    # input paths
    data_folder_path = "../resources/bbcsport/"
    stop_file_path = "../resources/stopword-list.txt"

    # output paths
    json_file_path = "../out/json_out.json"
    vocab_file_path = "../out/vocab.txt"
    class_file_path = "../out/class-tf.json"
    train_file_path = "../out/train-set.json"
    test_file_path = "../out/test-set.json"

    labels = os.listdir(data_folder_path)  # labels = folders

    ser = Serialization()
    stop_list = ser.importStopList(stop_file_path)
    ser.preprocessing.stop_word = stop_list
    json_list = ser.readRawData(data_folder_path)
    ser.shuffleJSONObjects(json_list)
    ser.writeToJSONFile(json_list, json_file_path)
    # ser.sortClassTerms()
    # ser.writeToJSONFile(ser.class_terms, class_file_path)

    # insert feature selection here and uncomment class_terms
    # feature selection should return same DS as json_list

    # splits data into 70/30 ratio
    train_set, test_set = dataSplit(json_list)

    ser.writeToJSONFile(train_set, train_file_path)
    ser.writeToJSONFile(test_set, test_file_path)

    vocabulary = createVocabulary(train_set)
    vocabulary_len = len(vocabulary)



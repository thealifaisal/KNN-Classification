# ******************************************
#
#   AUTHOR: ALI FAISAL
#   Github: https://github.com/thealifaisal/
#
# ******************************************

import openpyxl
from datetime import datetime
from src.ml_vsm import MachineLearning
from src.serialization import Serialization

if __name__ == "__main__":

    print(datetime.now().strftime("%H:%M:%S") + ": setting paths...")
    # input paths
    data_folder_path = "../resources/bbcsport/"
    stop_file_path = "../resources/stopword-list.txt"

    # output paths
    json_file_path = "../out/json_out.json"
    vocab_file_path = "../out/vocab.txt"
    class_file_path = "../out/class-tf.json"
    train_file_path = "../out/train-set.json"
    test_file_path = "../out/test-set.json"
    train_vectors_path = "../out/train-tf-idf.xlsx"

    print(datetime.now().strftime("%H:%M:%S") + ": serializing raw data...")
    ser = Serialization()
    # imports stoplist
    stop_list = ser.importStopList(stop_file_path)
    ser.preprocessing.stop_word = stop_list

    # returns a serialized data from raw text files
    # e.g: json_list = [{"id": id1, "label": lb, "features": {"term": tf}}, {...}, {...}, ...]
    json_list = ser.readRawData(data_folder_path)

    # randomize all the files for fair splitting
    ser.shuffleJSONObjects(json_list)
    # ser.writeToJSONFile(json_list, json_file_path)

    print(datetime.now().strftime("%H:%M:%S") + ": splitting serialized data...")
    ml = MachineLearning()
    # splits data into 70/30 ratio
    train_set, test_set = ml.dataSplit(json_list)
    trainset_len = len(train_set)
    testset_len = len(test_set)

    print(datetime.now().strftime("%H:%M:%S") + ": lengths => train-data: " + str(trainset_len)
          + ", test-data: " + str(testset_len))

    # class_terms = ser.classTermFrequency(train_set)
    # ser.sortClassTerms(class_terms)
    # ser.writeToJSONFile(class_terms, class_file_path)

    # insert feature selection here and uncomment class_terms
    # input: train_set, class_terms
    # output: train_set (with relevant features)

    print(datetime.now().strftime("%H:%M:%S") + ": writing to json files...")
    ser.writeToJSONFile(train_set, train_file_path)
    ser.writeToJSONFile(test_set, test_file_path)

    print(datetime.now().strftime("%H:%M:%S") + ": creating vocabulary...")
    vocabulary = ml.createVocabulary(train_set)
    vocabulary_len = len(vocabulary)
    print(datetime.now().strftime("%H:%M:%S") + ": vocabulary size = " + str(vocabulary_len))

    wb = openpyxl.Workbook()
    wb_sheet = wb.active

    print(datetime.now().strftime("%H:%M:%S") + ": preparing xlsx sheet...")
    ml.prepareSheet(wb_sheet, train_set)

    print(datetime.now().strftime("%H:%M:%S") + ": creating training vectors...")
    ml.createTrainVectors(vocabulary, wb_sheet, train_set)

    # print(datetime.now().strftime("%H:%M:%S") + ": saving training vectors...")
    # wb.save(train_vectors_path)

    print(datetime.now().strftime("%H:%M:%S") + ": started testing...")

    # number of correct predictions
    correct_predictions = 0

    # iterating for every test file
    for i in range(0, testset_len):
        test_obj = test_set[i]
        ml.createTestVector(vocabulary, wb_sheet, test_obj, trainset_len)

        # returns a result dict, where doc with highest cosine is the first item
        # result_set = {"062_c": 0.51468, ...}
        result_set = ml.cosineSimilarity(trainset_len, vocabulary_len, wb_sheet)

        # k = 3; to select top k neighbors/docs from result_set
        k = 3

        # e.g: test_composite_key = 096_c
        test_composite_key = test_obj["id"] + "_" + test_obj["label"][0]
        correct_predictions += ml.classifyKNN(result_set, k, test_composite_key)

    # accuracy = no of files correctly predicted / total files tested
    accuracy = format(correct_predictions / testset_len, ".5f")

    print(datetime.now().strftime("%H:%M:%S") + ": KNN accuracy = " + accuracy)

    print(datetime.now().strftime("%H:%M:%S") + ": detailed result saved in ../out/predictions.txt")

    # ----------- the end -----------

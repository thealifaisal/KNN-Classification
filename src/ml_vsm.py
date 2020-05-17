# ******************************************
#
#   AUTHOR: ALI FAISAL
#   Github: https://github.com/thealifaisal/
#
# ******************************************

import math
import operator


class MachineLearning:

    @staticmethod
    def createTrainVectors(vocabulary, wb_sheet, train_set):
        # train_set = [{id, label, features: {term: tf}}, ]
        trainset_len = len(train_set)
        # iterates for every vocabulary
        for i, voc in enumerate(vocabulary):
            wb_sheet.cell(i + 2, 1).value = voc
            df = 0
            for doc in range(0, trainset_len):
                try:
                    # assigns tf of term in doc
                    wb_sheet.cell(i + 2, doc + 2).value = train_set[doc]["features"][voc]
                    df += 1
                except KeyError:
                    # when, term from vocab not in doc
                    wb_sheet.cell(i + 2, doc + 2).value = 0

            idf = float(format(math.log10(trainset_len / df), ".5f"))
            wb_sheet.cell(i + 2, trainset_len + 2).value = idf

            for doc in range(0, trainset_len):
                tf = float(format(wb_sheet.cell(i + 2, doc + 2).value, ".5f"))
                wb_sheet.cell(i + 2, doc + 2).value = tf * idf
        return

    @staticmethod
    def createTestVector(vocabulary, wb_sheet, test_doc, trainset_len):
        for i, voc in enumerate(vocabulary):

            try:
                tf = test_doc["features"][voc]
            except KeyError:
                tf = 0

            idf = float(wb_sheet.cell(i + 2, trainset_len + 2).value)
            wb_sheet.cell(i + 2, trainset_len + 3).value = format(tf * idf, ".5f")
        return

    @staticmethod
    def prepareSheet(wb_sheet, train_set):
        # train_set = [{id, label, features: {term: tf}}, {}, ...]
        trainset_len = len(train_set)
        wb_sheet.cell(1, 1).value = "bag"
        for _i in range(2, trainset_len + 1):
            # e.g: doc-id = 037a, 110t, ...
            wb_sheet.cell(1, _i).value = train_set[_i - 2]["id"] + "_" + train_set[_i - 2]["label"][0]
        wb_sheet.cell(1, trainset_len + 2).value = "idf"
        return

    # splits the data-set into train and test sets
    # returns trains and test sets
    @staticmethod
    def dataSplit(jsonlist):
        # jsonlist = [{id: id1, label: lb1, features: {term: tf}}, {}, {}, ...]

        docs_len = len(jsonlist)  # 737
        train_len = round(70 / 100 * docs_len)  # 516
        test_len = docs_len - train_len  # 221
        trainset = []
        testset = []

        for f in range(0, train_len):
            trainset.append(jsonlist[f].copy())

        for f in range(train_len, docs_len):
            testset.append(jsonlist[f].copy())

        jsonlist.clear()

        return trainset, testset

    @staticmethod
    def createVocabulary(trainset):
        vocab = []
        for file in trainset:
            for feature in file["features"].keys():
                if feature not in vocab:
                    vocab.append(feature)
        return vocab

    @staticmethod
    def cosineSimilarity(trainset_len, vocabulary_len, wb_sheet):
        result_set = {}

        for i in range(0, trainset_len):

            train_doc_col = i + 2
            test_doc_col = trainset_len + 3
            dot_product = 0
            doc_sum = 0
            test_sum = 0

            for j in range(0, vocabulary_len):
                doc_tf_idf = float(wb_sheet.cell(j + 2, train_doc_col).value)
                test_tf_idf = float(wb_sheet.cell(j + 2, test_doc_col).value)
                dot_product += (doc_tf_idf * test_tf_idf)
                doc_sum += (doc_tf_idf ** 2)
                test_sum += (test_tf_idf ** 2)

            try:
                cosine = dot_product / (math.sqrt(doc_sum) * math.sqrt(test_sum))
            except ZeroDivisionError:
                cosine = -1

            if cosine != -1:
                composite_key = wb_sheet.cell(1, train_doc_col).value
                result_set[composite_key] = cosine

        result_set = dict(sorted(result_set.items(), key=operator.itemgetter(1), reverse=True))

        cosines_file = open("../out/cosines.txt", "w")
        cosines_file.write(str(result_set))
        cosines_file.close()

        return result_set

    @staticmethod
    def classifyKNN(result_set, k, test_composite_key):
        # KNN:
        #   key: doc-class
        #   value: number of docs that belong to that class from top K result_set
        # e.g: KNN = {"a": 2, "t": 1}
        # e.g: KNN = {"a": 1`, "t": 1, "r": 1}
        KNN = {}
        labels = {"a": "athletics", "c": "cricket", "f": "football", "r": "rugby", "t": "tennis"}

        # i=0, 1, 2, .. | item=("067_c", 0.12458), ...
        for i, item in enumerate(result_set.items()):

            if i >= k:
                # after selecting top k, break
                break

            # item = ("003_a", 0.12312)
            # item[0]="003_a", item[1]=0.12312
            # letter = a, c, f, r, t
            letter = item[0].split("_")[1]

            # if a class of doc from top K result is not in KNN
            if letter not in KNN.keys():
                # insert class with its initial doc occurrence = 1
                KNN[letter] = 1
            else:
                # else increment doc occurrence from that class
                KNN[letter] += 1

        # sort KNN in descending order,
        # so that class with most docs in top K results is at starting index
        # e.g: KNN = {"a": 2, "t": 1} -> 2 docs are from class 'a', 1 doc is from class 't'
        KNN = dict(sorted(KNN.items(), key=operator.itemgetter(1), reverse=True))

        # fetch item with max doc occurrences from the start
        # e.g: max_item = ("a": 2)
        max_item = list(KNN.items())[0]

        # check if all values are same in KNN
        #   means that in top K results, each class occurred has same number of docs occurrences
        # e.g: KNN = {"a": 1`, "t": 1, "r": 1}
        # all_values_same = {True, False}
        all_values_same = all(x == max_item[1] for x in KNN.values())

        if all_values_same:
            # then select the doc from result_set with max cosine value
            # e.g: doc_key = "003_a"
            doc_key = list(result_set.keys())[0]
            predicted_class = doc_key.split("_")[1]
        else:
            # otherwise select the class with most documents in KNN
            # e.g: doc_key = "003_a"
            doc_key = max_item[0]
            predicted_class = doc_key.split("_")[1]

        actual_class = test_composite_key.split("_")[1]

        file = open("../out/prediction.txt", "a+")

        if predicted_class == actual_class:
            file.write("test_file: " + test_composite_key + ", predicted class: " + labels[predicted_class] + "\n")
            correct = 1
        else:
            file.write("test_file: " + test_composite_key + ", predicted class: " + labels[predicted_class] + "\n")
            correct = 0

        return correct

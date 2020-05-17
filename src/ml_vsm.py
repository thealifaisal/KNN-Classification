import math


class MachineLearning:

    def createTrainVectors(self, vocabulary, wb_sheet, train_set):
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

    def prepareSheet(self, wb_sheet, train_set):
        # train_set = [{id, label, features: {term: tf}}, {}, ...]
        trainset_len = len(train_set)
        wb_sheet.cell(1, 1).value = "bag"
        for _i in range(2, trainset_len + 1):
            # e.g: doc-id = 037a, 110t, ...
            wb_sheet.cell(1, _i).value = train_set[_i - 2]["id"] + train_set[_i - 2]["label"][0]
        wb_sheet.cell(1, trainset_len + 2).value = "idf"
        return

    # splits the data-set into train and test sets
    # returns trains and test sets
    def dataSplit(self, jsonlist):
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

    def createVocabulary(self, trainset):
        vocab = []
        for file in trainset:
            for feature in file["features"].keys():
                if feature not in vocab:
                    vocab.append(feature)
        return vocab
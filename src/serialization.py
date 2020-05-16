import os
import json
from random import shuffle
from src.pre_processing import Preprocessing
import operator


class Serialization:

    # constructor
    def __init__(self):
        self.preprocessing = Preprocessing()
        self.class_terms = {}

    # reads raw text files and serialize them to JSON format
    # returns a list
    def readRawData(self, main_data_path):
        jsonlist = []
        folders = os.listdir(main_data_path)
        # main_data_path = bbcnews
        # folders = folders in the main directory as list
        for folder in folders:
            # files = files in the sub-folder as list
            files = os.listdir(main_data_path + folder + "/")

            for file in files:
                # for each file, a JSON object is created
                json_str = self.createJSONString(main_data_path + folder + "/" + file)
                # JSON object is added in the list
                jsonlist.append(json_str)
        return jsonlist

    def createJSONString(self, file_path):

        file = open(file_path, "r")

        tokens = self.preprocessing.tokenizer(file.read() + " ")
        lemmas = self.preprocessing.lemmatizer(tokens)

        # file_path = "../resources/bbcsport/tennis/034.txt"
        ID = file_path.split("/")[4].split(".")[0]
        label = file_path.split("/")[3]
        json_string = {
            "id": ID,
            "label": label,
            "features": lemmas
        }
        # self.classTermFrequency(lemmas, label)

        file.close()
        return json_string

    # shuffles a JSON list so that
    # there is no bias in splitting the data set into train/test
    def shuffleJSONObjects(self, jsonlist):
        shuffle(jsonlist)
        return

    # writes the data(list or dict) in to JSON file
    def writeToJSONFile(self, data, jsonfile_path):

        # opens a JSON file for writing
        file = open(jsonfile_path, "w")

        # checks the ype of data
        if type(data) is list:
            # if list, then iterates for every object in list
            # e.g: [{id, label, features}, {...}, ...]
            for obj in data:
                # obj = {id, label, features}
                json_object = json.dumps(obj, indent=4)
                file.write(json_object + "\n")

        elif type(data) is dict:
            # if dictionary
            # e.g: {class-1: {term-1: tf, term-2: tf}, class-2: {...}, ...}
            json_object = json.dumps(data, indent=4)
            file.write(json_object + "\n")

        file.close()
        return

    def sortClassTerms(self):
        # source: https://www.w3resource.com/python-exercises/dictionary/python-data-type-dictionary-exercise-1.php
        # noinspection PyTypeChecker
        for ct in self.class_terms:
            self.class_terms[ct] = dict(sorted(self.class_terms[ct].items(), key=operator.itemgetter(1),
                                               reverse=True))
        return

    # lemmas = terms in a file as keys and term-frequencies as values
    # label = label/class to which the file belongs
    def classTermFrequency(self, lemmas, label):

        # iterates for every term in document
        for lem in lemmas.keys():

            # class_terms = {class-1: {term-1: tf}, class-2: {}, ...}
            # checks if label/class is in the dict
            if label not in self.class_terms.keys():
                # if not, inserts a label with empty dict
                self.class_terms[label] = {}

            # checks if term is in a particular class
            if lem not in self.class_terms[label].keys():
                # if not, adds a the term in class with its term frequency of current doc
                self.class_terms[label][lem] = lemmas.get(lem)
            else:
                # if exists, fetches the tf for the term in class
                # fetches the tf for the term in the current doc
                # sums the two tf, and assigns this new tf against the term in class
                self.class_terms[label][lem] = self.class_terms[label].get(lem) + lemmas.get(lem)
        return

    def importStopList(self, path):
        # a stop-word file is opened and parsed to be saved as a list
        # when path is not empty
        if path != "":
            try:
                # try opening a file
                stop_file = open(path, "r")
                # when opens, parse the file and save stop-words to list
                stoplist = stop_file.read().split("\n")
                # and close the file
                stop_file.close()
            except FileNotFoundError:
                # if file not opens, an empty list is returned
                stoplist = []
        else:
            # if path is empty, an empty list is returned
            stoplist = []

        # returning a list
        return stoplist

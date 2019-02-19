import spotlight
import glob
from subprocess import Popen, STDOUT, PIPE
import os
import json
import re
from config import Config
from gcloud import upload_to_cloud_storage
from random import random
from time import sleep
import progressbar
import requests

def file_len(file_path):
    with open(file_path) as f:
        if f:
            for i, l in enumerate(f):
                pass
            return i + 1
        else:
            return 0

def retrieve_uri(input_path, output_path, url=Config.DBPEDIA_SPOTLIGHT, confidence=0.5,
                 support=0, verbose=False, file_limit=None):
    """
    Step 1: Use DBPedia Spotlight to retrieve entities from text file. The two required parameters are `input_path` and
    `output_path`

    :param input_path: Path leading to text files to analyse. Format 'path/*.txt'
    :param output_path: Path in which output should be saved. Format 'path/'
    :param url: URL at which the API can be called. Default is the demo API.
    :param confidence: Setting a high confidence threshold instructs DBpedia Spotlight to avoid incorrect annotations
    as much as possible at the risk of losing some correct ones. Integer between 0 and 1. Default is the recommended
    parameter
    :param support: minimum number of inlinks a DBpedia resource has to have in order to be annotated.
    Integer between 0 and 1000.
    :param verbose: print debugging information
    :param file_limit: limit the number of files to process
    :return: None
    """
    # List of files to go through
    files = glob.glob(input_path)[:file_limit]
    lines = sum([file_len(f_path) for f_path in files])

    # Populate the list with files that have already been processed
    processed_files = []
    for processed_file in glob.glob(output_path+'*.txt'):
        processed_files.append(processed_file.split("/")[-1].split(".")[0])

    if verbose:
        print("Processing files:")
    else:
        # Progress bar
        bar = progressbar.ProgressBar(maxval=lines,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
    i = 0
    # Process each file in the input path one by one
    for input_file_path in files:
        filename = input_file_path.split("/")[-1].split(".")[0]
        if verbose:
            print(filename)
        if filename not in processed_files:
            with open(input_file_path, 'r') as input_file:
                # data is a list.
                # Each element of the list is a String corresponding to one line in the original file.
                data = list(input_file.readlines())
            tags = []
            for line in data:
                # Remove unprintable XML character that produces error when calling the API
                line = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', line)
                try:
                    annotations = spotlight.annotate(url, line, confidence=confidence, support=support)
                except spotlight.SpotlightException:
                    annotations = []
                    if verbose:
                        print(line)
                except requests.exceptions.HTTPError:
                    print(f"Error with :'{line}'")


                for annotation in annotations:
                    # Entity type
                    uri = annotation.get("URI")
                    # Starting index in text
                    start_idx = annotation.get("offset")
                    # Entity form in the text
                    word = annotation.get("surfaceForm")
                    end_idx = start_idx + len(word)
                    # Get or create the parent type
                    if annotation.get("types"):
                        list_of_types = [elt for elt in annotation["types"].split(',') if "DBpedia" in elt]
                        type_ = list_of_types[0]
                    else:
                        type_ = "ex:" + uri.split("/")[-1]
                    tags.append(json.dumps((word, uri, type_, start_idx, end_idx)))

                if url == "https://api.entity_resolution-spotlight.org/en/annotate?":
                    # If using the demo API, wait between each file to avoid getting banned
                    sleep(random() * 4)
                if not verbose:
                    i += 1
                    bar.update(i)

            output_file_path = output_path + filename + ".txt"
            with open(output_file_path, 'w') as output_file:
                output_file.write("\n".join(tags))


    if not verbose:
        bar.finish()


def stanford_coref(input_path, output_path, path_to_stanford_jar, verbose=False, file_limit=None):
    """
    Step2: extract coreferences from text using Stanford CoreNLP API.

    :param input_path: Path leading to text files to analyse. Format 'path/*.txt'
    :param output_path: Path in which output should be saved. Format 'path/'
    :param path_to_stanford_jar: Path leading to the Stanford CoreNLP files. Format 'path/'
    :param verbose: print debugging information
    :param file_limit: Max number of files to process
    :return: None
    """
    # Populate list with already processed files
    processed_files = []
    for processed_file in glob.glob(output_path + '*.json'):
        processed_files.append(processed_file.split("/")[-1].split(".")[0])

    # Create file with names of files to process
    files_to_process = []
    for file_name in glob.glob(input_path)[:file_limit]:
        if file_name.split('/')[-1].split(".")[0] not in processed_files:
            files_to_process.append("../" + file_name)
    with open(output_path+"files_to_process.txt", "w") as filelist:
        filelist.write("\n".join(files_to_process))

    if not verbose:
        bar = progressbar.ProgressBar(maxval=len(files_to_process),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

    # Command to run to launch the CoreNLP API
    command = f'cd {path_to_stanford_jar}; java -mx8g ' \
              f'-cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP ' \
              f'-annotators tokenize,ssplit,pos,lemma,ner,parse,mention,coref -tokenize.whitespace ' \
              f'-coref.algorithm neural -ssplit.eolonly ' \
              f'-coref.maxMentionDistance 30 -filelist "{"../"+output_path+"files_to_process.txt"}" -outputFormat json ' \
              f'-outputDirectory {"../"+output_path} ' \
              f'-replaceExtension'
    if verbose:
        print('Executing command = {}'.format(command))
        # java_process = Popen(command, stdout=stderr, shell=True)
        java_process = Popen(command, shell=True)
    else:
        i = 0
        java_process = Popen(command, stderr=STDOUT, stdout=PIPE, shell=True)
        for line in iter(java_process.stdout.readline, b''):
            line = line.rstrip().decode('utf-8')
            if 'Annotating file' in line:
                i += 1
                bar.update(i)
        bar.finish()

    java_process.wait()
    assert not java_process.returncode, 'ERROR: Call to stanford_coref exited with a non-zero code status.'

    os.remove(output_path + 'files_to_process.txt')


def extract_coreferences(input_path, output_path, verbose=False):
    """
    Clean output of step 2. Tranform json files to text files by removing useless information.

    :param input_path: path in which to read files. Format 'path/*.json'
    :param output_path: path in which to save cleaned files. Format 'path/'.
    :return: None
    """
    for file_path in glob.glob(input_path):
        # For each json file
        filename = file_path.split("/")[-1].split(".")[0]
        with open(file_path, 'r') as coref_file:
            output = json.load(coref_file)

        entities = []
        # Populate list with entities
        # One element will have the following format
        # (entity, start index, end index)
        for i in output['corefs'].keys():
            if len(output['corefs'][i]) > 2:
                entity = []
                set_entity = set()
                for ref in output['corefs'][i]:
                    start_index = "NAN"
                    if ref['text'] not in set_entity:
                        for sentence in output['sentences']:
                            if sentence['line'] == ref['sentNum']:
                                for token in sentence['tokens']:
                                    if token['index'] == ref['startIndex']:
                                        start_index = token['characterOffsetBegin']
                                    if token['index'] == ref['endIndex']:
                                        end_index = int(token['characterOffsetBegin']) - 1
                                        break
                        try:
                            entity.append((ref['text'], start_index, end_index))
                        except:
                            print(ref['text'], ref['sentNum'])
                        set_entity.add(ref['text'])
                entities.append(entity)
        with open(output_path+filename+".txt", 'w') as cleaned_file:
            cleaned_file.write("\n".join([json.dumps(elt) for elt in entities]))


def stanford_ie(input_path, output_path, path_to_stanford_jar, verbose=False, file_limit=None):
    """
    Step 3: Extract relation triples in text (subject, relation, object).

    :param input_path: Path leading to text files to analyse. Format 'path/*.txt'
    :param output_path: Path in which output should be saved. Format 'path/'
    :param path_to_stanford_jar: Path leading to the Stanford CoreNLP files. Format 'path/'
    :param verbose: print debugging information
    :param file_limit: Max number of files to process
    :return: None
    """
    # Populate list with processed files
    processed_files = []
    for processed_file in glob.glob(output_path + '*.json'):
        processed_files.append(processed_file.split("/")[-1].split(".")[0])

    # Create file with names of files to process
    files_to_process = []
    for file_name in glob.glob(input_path)[:file_limit]:
        if file_name.split('/')[-1].split(".")[0] not in processed_files:
            files_to_process.append("../" + file_name)
    with open(output_path+"files_to_process.txt", "w") as filelist:
        filelist.write("\n".join(files_to_process))

    if not verbose:
        bar = progressbar.ProgressBar(maxval=len(files_to_process),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

    # Java command to run to launch Stanford CoreNLP API
    command = f'cd {path_to_stanford_jar}; java -mx8g ' \
              f'-cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP ' \
              f'-annotators tokenize,ssplit,pos,lemma,depparse,natlog,openie -tokenize.whitespace -ssplit.eolonly ' \
              f'-filelist "{"../"+output_path+"files_to_process.txt"}" -outputFormat json ' \
              f'-outputDirectory {"../"+output_path} ' \
              f'-replaceExtension'

    if verbose:
        print('Executing command = {}'.format(command))
        # java_process = Popen(command, stdout=stderr, shell=True)
        java_process = Popen(command, shell=True)
    else:
        i = 0
        java_process = Popen(command, stderr=STDOUT, stdout=PIPE, shell=True)
        for line in iter(java_process.stdout.readline, b''):
            line = line.rstrip().decode('utf-8')
            if 'Annotating file' in line:
                i += 1
                bar.update(i)
        bar.finish()

    java_process.wait()
    assert not java_process.returncode, 'ERROR: Call to stanford_ie exited with a non-zero code status.'

    os.remove(output_path+'files_to_process.txt')


def clean_open_ie(input_path, output_path):
    """
    Clean output of step 3. Tranform json files to text files by removing useless information.

    :param input_path: path in which to read files. Format 'path/*.json'
    :param output_path: path in which to save cleaned files. Format 'path/'.
    :return: None
    """
    for file in glob.glob(input_path):
        filename = file.split("/")[-1].split(".")[0]
        relations = []
        with open(file, 'r') as raw_output_file:
            output = json.load(raw_output_file)

        # Output Format:
        # [(Subject, Index Start), (Relation, Index Start), (Object, Index Start)]
        for sentence in output['sentences']:
            for relation in sentence['openie']:
                subject, subject_word_idx = relation['subject'], relation['subjectSpan'][0] + 1
                link, link_word_idx = relation['relation'], relation['relationSpan'][0] + 1
                object, object_word_idx = relation['object'], relation['objectSpan'][0] + 1

                indexes = {subject_word_idx: 0, link_word_idx: 0, object_word_idx: 0}
                for token in sentence['tokens']:
                    if token['index'] in indexes.keys():
                        indexes[token['index']] = token['characterOffsetBegin']
                relations.append([(subject, indexes[subject_word_idx]), (link, indexes[link_word_idx]),
                                  (object, indexes[object_word_idx])])

        with open(output_path+filename+".txt", 'w') as output_file:
            output_file.write("\n".join([json.dumps(elt) for elt in relations]))


if __name__ == '__main__':
    TEXT_DATA_PATH = "text-files"
    TRIPLES = "triples/"
    if not os.path.isdir(TRIPLES):
        os.mkdir(TRIPLES)

    print("Step 1....")
    if not os.path.isdir(TRIPLES+"step1/"):
        os.mkdir(TRIPLES + "step1")
    retrieve_uri(TEXT_DATA_PATH, "triples/step1/", file_limit=2)
    upload_to_cloud_storage(TRIPLES+'step1/*.txt', 'step1/')
    print("Done!\n")

    print("Step 2....")
    if not os.path.isdir(TRIPLES+"step2/"):
        os.mkdir(TRIPLES + "step2")
    stanford_coref(TEXT_DATA_PATH, "triples/step2/", Config.STANFORD_CORE_NLP, file_limit=2)
    print("Cleaning....")
    extract_coreferences(TRIPLES  + 'step2/*.json', "triples/step2/")
    print("Uploading....")
    upload_to_cloud_storage(TRIPLES + 'step2/*.txt', 'step2/')
    print("Done!\n")

    print("Step 3....")
    if not os.path.isdir(TRIPLES+"step3/"):
        os.mkdir(TRIPLES+"step3/")
    stanford_ie(TEXT_DATA_PATH, "triples/step3/", Config.STANFORD_CORE_NLP, file_limit=2)
    print("Cleaning...")
    clean_open_ie(TRIPLES + "step3/*.json", TRIPLES + "step3/")
    print("Uploading...")
    upload_to_cloud_storage(TRIPLES + 'step3/*.txt', 'step3/')
    print("Success!!")

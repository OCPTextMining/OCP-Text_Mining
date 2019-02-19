from gcloud import upload_to_cloud_storage
from extract_text import parse_files, clean_files
from triples_extraction import retrieve_uri, extract_coreferences, stanford_coref, stanford_ie, clean_open_ie
from triples_integration import triples_integration
import argparse
import os
from config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a folder containing files to extract '
                                                 'information into triples.')
    parser.add_argument('-f', dest='path', help='Path to files to process', required=True)
    parser.add_argument('-p', dest='pdf', action='store_true',
                        default=False,
                        help='Input is PDF files')
    parser.add_argument('-l', dest='limit', help='Maximum number of files to process')
    parser.add_argument('-debugging', dest='debug', help='Print debugging information', action='store_true')

    args = parser.parse_args()
    INPUT_PATH = args.path
    TRIPLES_PATH = 'triples/'

    if args.pdf:
        TEXT_PATH = 'text-files/*.txt'
        parse_files(args.path, TEXT_PATH, int(args.limit))
        clean_files(TEXT_PATH)
        upload_to_cloud_storage(TEXT_PATH, 'text-files/')
        INPUT_PATH = TEXT_PATH

    # Create Triples directory
    if not os.path.isdir(TRIPLES_PATH):
        os.mkdir(TRIPLES_PATH)

    print("Extract Triples\n")
    print("Step 1....")
    if not os.path.isdir(TRIPLES_PATH+"step1/"):
        os.mkdir(TRIPLES_PATH + "step1")
    retrieve_uri(INPUT_PATH, TRIPLES_PATH + "step1/", file_limit=int(args.limit), verbose=args.debug)
    print("Uploading...")
    upload_to_cloud_storage(TRIPLES_PATH+'step1/*.txt', 'step1/')
    print("Done!\n")

    print("Step 2....")
    if not os.path.isdir(TRIPLES_PATH+"step2/"):
        os.mkdir(TRIPLES_PATH + "step2")
    stanford_coref(INPUT_PATH, TRIPLES_PATH + "step2/", Config.STANFORD_CORE_NLP, file_limit=int(args.limit),
                   verbose=args.debug)
    print("Cleaning....")
    extract_coreferences(TRIPLES_PATH + 'step2/*.json', TRIPLES_PATH + "step2/")
    print("Uploading....")
    upload_to_cloud_storage(TRIPLES_PATH + 'step2/*.txt', 'step2/')
    print("Done!\n")

    print("Step 3....")
    if not os.path.isdir(TRIPLES_PATH+"step3/"):
        os.mkdir(TRIPLES_PATH+"step3/")
    stanford_ie(INPUT_PATH, TRIPLES_PATH + "step3/", Config.STANFORD_CORE_NLP, file_limit=int(args.limit),
                verbose=args.debug)
    print("Cleaning...")
    clean_open_ie(TRIPLES_PATH + "step3/*.json", TRIPLES_PATH + "step3/")
    print("Uploading...")
    upload_to_cloud_storage(TRIPLES_PATH + 'step3/*.txt', 'step3/')
    print("Done!")

    print("Integrate Triples...")
    triples_integration(TRIPLES_PATH)

import pdfminer.high_level
import pdfminer.layout
from gcloud import download_raw_files
import progressbar
import glob
import re
import os


laparams = pdfminer.layout.LAParams()
# laparams = None


def parse_files(raw_path, output_path, max_nb_files=10, verbose=False):
    already_parsed = []
    for parsed_file_path in glob.glob(output_path):
        already_parsed.append(parsed_file_path.split("/")[-1].split(".")[0])

    directory = raw_path.split("/")[0] + '/'
    if not os.path.isdir(directory):
        os.mkdir(directory)
        download_raw_files(directory)

    output_directory = output_path.split('/')[0] + '/'
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    files = glob.glob(raw_path)

    if not verbose:
        print("Extracting text from pdf")
        # Progress bar
        bar = progressbar.ProgressBar(maxval=len(files),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

    for i, input_file_path in enumerate(files):
        if verbose:
            print(input_file_path.split("/")[-1].split(".")[0].replace(" ", "_"))
        if input_file_path.split("/")[-1].split(".")[0].replace(" ", "_") not in already_parsed:
            output_folder = "/".join(output_path.split("/")[:-1])
            filename = input_file_path.split("/")[-1].split(".")[0]+".txt"
            output_file_path = "/".join([output_folder] + [filename.replace(" ", "_")])

            with open(input_file_path, "rb") as input_file:
                with open(output_file_path, "wb") as output_file:
                    try:
                        pdfminer.high_level.extract_text_to_fp(input_file, output_file, laparams=laparams)
                    except ValueError:
                        print(f"Unable to extract text from {input_file_path}")

            if i + 1 >= max_nb_files:
                break
        if not verbose:
            i += 1
            bar.update(i)

    if not verbose:
        bar.finish()


def clean_files(output_path):
    for file in glob.glob(output_path):
        with open(file, 'r', encoding="utf-8", errors="ignore") as raw_text:
            # Remove footer
            footer = "Phosphates Weekly Report"
            content = ''.join([elt for elt in raw_text.readlines() if footer not in elt])

            content = content.replace("\n","")
            # Remove useless line breaks
            # while content.find("\n\n") != -1 or content.find("\n \n") != -1:
            #     content = content.replace("\n\n", "\n")
            #     content = content.replace("\n \n", "\n")
            # Remove footer
            content = content.replace("www.crugroup.com ", "")
            # Remove non ASCII characters
            content = re.sub(r'[^\x00-\x7F]+',' ', content)

            # One sentence per line
            content = ".\n".join([elt.strip() for elt in content.split(". ")])
            # Do not write file if only 1 long line of text
            content = content if len(content.split(".\n")) > 1 else ""

        with open(file, 'w') as cleaned_text:
            title_position = content.find("ANALYSIS")
            start_position = title_position if title_position != -1 else 0
            content = content[start_position:]
            cleaned_text.write(content)


if __name__ == "__main__":
    parse_files('raw_data/*.pdf', 'text-files/*.pdf', 5)
    clean_files('text-files/*.txt')

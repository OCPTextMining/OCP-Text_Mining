# OCP Text Mining Project

## Requirements
- Python 3.6
- PDFminer.six (PDFminer for Python 3.x)
- Stanford Named Entity Recognition
- DBPedia Spotlight

## How to Setup
### Create a virtual environment
Create the environment
```
python -m venv venv
```
Activate the envrionment
```
source venv/bin/activate
```
You are now operating inside the virtual environment named `venv`. Every package you install will then only be added
to this environment and thus won't interfere with the main Python installation.

### PDFminer.six
On macOS, you should to install this module from source to avoid to avoid problem with line endings.
```
git clone https://github.com/pdfminer/pdfminer.six
```
Then, inside the newly created directory, run
```
python setup.py install
```

### DBPedia Spotlight
Download java models using
```
wget https://sourceforge.net/projects/dbpedia-spotlight/files/spotlight/dbpedia-spotlight-0.7.jar/download
wget https://sourceforge.net/projects/dbpedia-spotlight/files/2016-10/en/model/en.tar.gz/download
tar xzf en.tar.gz
```

### Stanford NER
Download the folder [here](https://nlp.stanford.edu/software/CRF-NER.html) and copy it to the root folder.

### Install required libraries
Use `pip install -r requirements.txt` to install all the required Python libraries.

### Google Cloud Authorization
If running locally, you must [download a service account](https://cloud.google.com/iam/docs/creating-managing-service-account-keys)
key to link your local version of Python with the Google Cloud account.

If running on a Google Cloud local server, you do not have to do anything on this step.

### Config file
Fill in the `config.py` file located at the root folder with the right paths:
 - URL to call DBPEdia Spotlight
 - Path the Standord Core API executable
 - Path to the Google Cloud service account

## How to run
### DBPedia Spotlight
First, instanciate the local DBPedia Server by running 
```
java -jar dbpedia-spotlight-0.7.jar en_2+2 http://localhost:2222/rest
```

### Run the main Python Program
This main program will execute the following steps:
 0. (Optional) Extract text from PDF files
 1. Named Entity Recognition using DBPedia Spotligh
 2. Coreference Resolution using Stanford CoreAPI
 3. Information Extractor using Stanford CoreAPI
 4. Upload all final triples to Google Cloud

```
usage: main.py [-h] -f PATH [-p] [-l LIMIT] [-debugging]

Process a folder containing files to extract information into triples.

optional arguments:
  -h, --help  show this help message and exit
  -f PATH     Path to files to process
  -p          Input is PDF files
  -l LIMIT    Maximum number of files to process
  -debugging  Print debugging information
 ```
 
Examples:
 - `python main.py -f 'raw_data/*.pdf' -p -l 5`: to create triples from the first 5 
 pdf files located in the `raw-data/` folder
 - `python main.py -f 'text-files/*.txt'`: to create triple from all text files located in the `text-files` folder
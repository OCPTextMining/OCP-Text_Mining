class Config:
    # Path to Google Cloud credentials
    GCLOUD_CREDENTIALS = 'credentials/service-account.json'

    # URL to DBPedia Spotlight
    # Defaults to the Demo API available online
    DBPEDIA_SPOTLIGHT = "https://api.entity_resolution-spotlight.org/en/annotate?"
    # DBPEDIA_SPOTLIGHT = "http://localhost:2222/rest/annotate?"

    # Path to Stanford CoreNLP folder
    STANFORD_CORE_NLP = "stanford-corenlp-full-2016-10-31"

    # Path to Triples folder
    # i.e. 'triples/'
    TRIPLES = "triples/"
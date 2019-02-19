# What we did this week
### Text extraction from PDF
Improved text extraction to 
  1. get one sentence per line (`extract_text.py` file)
  2. remove non ASCII characters
  3. remove the PDF footers 

### Triples extraction
Following the paper from Kertkeidkachorn and Ichise, we implemented the first three steps 
to extract triples from text (`triples_extraction.py`file)
  - **Tagging**: using DBPedia Spotlight. This step is working fine as a demo for now. But, we 
      are currently using the API provided DBPedia Spotlight *as demo purposes*. We thus need to 
        deploy our own server to process the files in bulk.
  - **Coreference resolution**: using Stanford CoreNLP - `CorefAnnotator`. 
      If we use the latest CoreNLP version `3.9.1`, we got an error while running the `CorefAnnotator`:
        1. Either an `OutOfMemory` error when we didn't specified any argument
        2. Or an `java.lang.NullPointerException` when we specified the following arguments: `-tokenize.whitespace 
            -ssplit.eolonly`
  We got around this error by downgrading the CoreNLP version to `3.7.0`.
  - **Information extraction**: using Stanford CoreNLP - `Open Information Extraction`

Currently, a process initiated per file and per step. But, Stanford CoreNLP provides a way to pass a `fileList` attribute to speed up the process.

# To do next week

1. Update the triple extraction to process file in bulk
   1. Deploy our own DBPedia Spotlight Server
   2. Update Stanford CoreNLP command to process files in bulk
2. Finish cleaning each output file we got in the triple extraction process
3. Use those three steps to produce the final triples



# 10/9 Meeting

- In the entity recognition part, can we find a list of abreviations to correctly tag them?

- One difficulty that can happen. We won't have static edges. The relations between entities will be updated at one point. How can we find a way to integrate temporal relations?

- *Graphical Models for Probabilistic and Causal Reasoning*, Judea Pearl. Propagation on graphs. Graph shows possibilities (steady-state graph) and the user impose a reality. In output, we get the impact of that particular event happening. Thus, inferring from a particular event the consequences.

- Add general inputs on the field as a user, e.g. phosphate needs oil. How can we add input from Wikipedia? Using reports that describe an industry to get a steady-state graph?

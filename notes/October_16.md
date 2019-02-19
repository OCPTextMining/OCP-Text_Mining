# What we did this week
### Triples extraction
Improvements on the three steps
  - **Tagging**: We weren't able to deploy our own server due to RAM issues 
  (8 Go of RAM needs to be allocated to the process). So, we used the online API
  given for demo purposes. This is working fine but slowly as we do not want 
  to get banned
  - **Coreference resolution**: We got around the error we had last week 
  by downgrading the CoreNLP version to `3.7.0`. This version is surprisingly much
  faster too. But, we are using the deterministic coreference resolution (`dcoref` 
  annotator). This one is faster but doesn't give the best results. We might want
  to use the version based on neural networks later. But, again, we would need more
  computing power.
  - **Information extraction**: using Stanford CoreNLP - `Open Information Extraction`
We updated the cleaning of all the 3 output files to be able to reuse them for the 
next step.
We processed all the 166 text files from the first data source using the `filelist`
argument which is way faster than processing file one by one. The timing was 
approximately:
 - 30 minutes for Step 1 (using the Demo API)
 - 4 hours for Step 2
 - 3 hours for Step 3
 
### Triples integration
We started integrating all of these three steps to get our final triples. 
The process is the following (extracted from the paper):
 1. First, identical entities are grouped using coreferring 
 chains from the coreference resolution component. 
 2. Second, a representative for the group of coreferring entities 
 is selected by the voting algorithm. 
 3. Third, all entities belonging to the group in the relation triples are 
 replaced by the representative of their groups.
 4. Fourth, the relation of a relation triple is straightforwardly transformed 
 into a predicate by assigning a new URI.
 5. Finally, if an object of a relation triple is not an entity, it is left 
 as literal. After performing these processes, text triples are extracted 
 from natural language text.

We implemented step 4 and 5.
We started also exploring the structure of DBPedia to see how we can add our 
own triples to the graph.

# To do next week

1. Making progress on triples integration
2. Understanding how we can add custom triples to the DBPedia graph. 

# 10/16 Meeting
# What we did this week
### Triples extraction
Improvements on the second step
  - **Tagging**: We still are not able to deploy our own server due to RAM issues 
  (8 Go of RAM needs to be allocated to the process).
  - **Coreference resolution**: We switched from the deterministic coreference 
  annotator (`dcoref`) to the neural coreference annotator (`coref`).
  - **Information extraction**: Nothing changed
 
### Triples integration
We continued the implementation fo the triples integration. We implemented steps 1 to 5 
from the scientific paper 
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

So, we are now able, from the three triples extraction step to integrate them to get
the final triples.

But, one major issue remains: *compared with the examples given in the scientific 
paper on which we based our work, our triples are much more complicated*.
For instance, they are using simple triples like: `(Obama, born in, Hawaii)` where the
three elements (subject, relation, object) are clearly determined.

Our triples are more complicated. Take this sentence for instance:
"Jordan export price levels are indicatively assessed in the $430-445/mt FOB level"
The two identified entities are "Jordan" and "FOB" (Free on Board). The relation is "are assessed in".
The triple `(Jordan, are assessed in, FOB)` won't add any information. We need to add the adjective "export price"
to the object and "in the $430-445/mt" attribute to the subject of the relation. We also need to add the
date of the document as this information is only valid in a certain time period. And which price 
are we referring to? If look back into the original document, we see that this paragraph is related to 
DAP (Diammonium phosphate) prices.

Ideally, we would want to extract `(Jordan DAP price, (are assessed, 430-445), FOB)` with 430-445 being
an attribute of the relation.

### Exporting triples to the DBPedia graph
**TO COMPLETE**

# To do next week

1. Improving triple integration
2. Understanding how we can add custom triples to the DBPedia graph. 

# 10/23 Meeting
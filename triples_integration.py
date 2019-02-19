import os
import glob
import json
import progressbar
from config import Config
from gcloud import upload_to_cloud_storage


def find_related_triples_by_relation(triple, lines):
    print(triple)
    relation, idx = triple[0], triple[1]
    related_idx = []
    for i, line in enumerate(lines):
        object_comp, relation_comp, subject_comp = line[0], line[1], line[2]
        new_relation_comp, new_idx_comp = relation_comp
        if new_idx_comp == idx and relation == new_relation_comp:
            related_idx.append(i)
    return related_idx


def disambiguate(step2_path, step3_path, verbose=False):
    """
    Use step2 (coreference resolution) to disambiguate (remove pronouns) from triples created at step 3.

    :param step2_path: path leading to step 2
    :param step3_path: path leading to step 3
    :param verbose: debug
    :return: None (overwrite step 3 triples)
    """
    files = glob.glob(step2_path+"*.txt")
    if not verbose:
        bar = progressbar.ProgressBar(maxval=len(files),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
    for i, file_path in enumerate(files):
        # Retrieve file name from step 2 folder
        filename = file_path.split("/")[-1].split(".")[0]
        if verbose:
            print(filename)

        # Create dict with the following structure
        # (Pronoun, start_index): Primary key
        map_secondary_to_key = {}

        # Open the coreference file
        with open(step2_path+filename+".txt", 'r') as step2_file:
            for line in step2_file.readlines():
                similar_items = json.loads(line)

                for item in similar_items[1:]:
                    map_secondary_to_key[(item[0], item[1])] = similar_items[0][0]

        # Open the triples file
        with open(step3_path+filename+".txt", 'r') as step3_file:
            relations = []
            for j, line in enumerate(step3_file.readlines()):
                # Check every triple
                relation = json.loads(line)
                subject, subject_idx = relation[0][0], relation[0][1]
                object, object_idx = relation[2][0], relation[2][1]
                # If the triple (subject, start_index) is in the dictionary
                # it means that it's a pronoun that needs to be replaced
                if (subject, subject_idx) in map_secondary_to_key.keys():
                    relation[0] = (map_secondary_to_key[(subject, subject_idx)], subject_idx)
                    if verbose:
                        print(f"line {j} modified: {subject} ==> {map_secondary_to_key[(subject, subject_idx)]}")
                if (object, object_idx) in map_secondary_to_key.keys():
                    relation[2] = (map_secondary_to_key[(object, object_idx)], subject_idx)
                    if verbose:
                        print(f"line {j} modified: {object} ==> {map_secondary_to_key[(object, object_idx)]}")
                relations.append(relation)

        with open(step3_path+filename+".txt", 'w') as step3_file:
            # Overwrite step3 file with the disambiguated version
            step3_file.write("\n".join([json.dumps(elt) for elt in relations]))
        if not verbose:
            bar.update(i+1)
    if not verbose:
        bar.finish()


def remove_identical_triples(step3_path, verbose):
    """
    Remove duplicate triples coming from the OpenIE (step3). A lot of triples are just overlapping each other. This
    function removes the duplicates.
    For each triple, we first identify the duplicates and we then keep the maximal triple among the duplicated ones.

    :param step3_path: path leading to step 3
    :param verbose: Debug
    :return: None (overwrite files in step3_path)
    """
    files = glob.glob(step3_path + "*.txt")
    if not verbose:
        bar = progressbar.ProgressBar(maxval=len(files),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
    for i, file_path in enumerate(files):
        filename = file_path.split("/")[-1].split(".")[0]

        with open(step3_path+filename+".txt", 'r') as step3_file:
            lines = [json.loads(l) for l in step3_file.readlines()]

        # Set which will contains indices of duplicated triples
        removed_lines = set()
        # Set which will contains indices of the maximal triple
        to_keep = set()
        # Dictionary mapping one triple with the similar triples
        similar_triples = {}

        for j, triple in enumerate(lines):
            # We consider triple j only if it was not already considered as a duplicate
            if j not in removed_lines:
                # First, find the triples similar to triple j
                # Each triple has the structure
                # (Subject, Starting Index of the Subject), (Relation, Starting Index), (Object, Starting Index)
                sub, rel, obj = triple[0], triple[1], triple[2]
                sub_len, sub_idx = len(sub[0]), int(sub[1])
                rel_len, rel_idx = len(rel[0]), int(rel[1])
                obj_len, obj_idx = len(obj[0]), int(obj[1])
                # We initialize the similar_triple dictionary
                similar_triples[j] = [(j, sub_len + rel_len + obj_len)]

                # We initialize a set of indices.
                # sub_idx_set will contain all the starting indices of the subject of the similar triples
                sub_idx_set = {sub_idx}
                # same for the relation
                rel_idx_set = {rel_idx}
                # same for the object
                obj_idx_set = {obj_idx}
                if verbose:
                    print(f"Current ({j}): {sub}\t{rel}\t{obj}")
                for k, t in enumerate(lines):
                    # Test if triple k is similar to triple j
                    cur_sub, cur_rel, cur_obj = t[0], t[1], t[2]
                    cur_sub_len, cur_sub_idx = len(cur_sub[0]), int(cur_sub[1])
                    cur_rel_len, cur_rel_idx = len(cur_rel[0]), int(cur_rel[1])
                    cur_obj_len, cur_obj_idx = len(cur_obj[0]), int(cur_obj[1])
                    # We consider triple k similar to j if:
                    # its subject starting index is in the set of subject starting index of triples similar to j
                    # its relation starting index is in the set of relation starting index of triples similar to j
                    # its object starting index is in the set of object starting index of triples similar to j
                    if (cur_sub_idx in sub_idx_set) or (cur_rel_idx in rel_idx_set) or (cur_obj_idx in obj_idx_set):
                        sub_idx_set.add(cur_sub_idx)
                        rel_idx_set.add(cur_rel_idx)
                        obj_idx_set.add(cur_obj_idx)

                        similar_triples[j].append((k, cur_sub_len + cur_rel_len + cur_obj_len))
                        if verbose:
                            print(f"Similar ({cur_sub_len + cur_rel_len + cur_obj_len}): "
                                  f"{cur_sub}\t{cur_rel}\t{cur_obj}")
                # Among all the similar triple, we keep the maximal one (j.e. the one with the most characters)
                max_triple = max(similar_triples[j], key=lambda x: x[1])[0]

                to_keep.add(max_triple)
                to_remove = [idx for idx, len in similar_triples[j]]
                removed_lines.update(to_remove)
                if verbose:
                    print("\n\n")

        # Write new file
        new_file = []
        for j, triple in enumerate(lines):
            if j in to_keep:
                new_file.append(triple)
        with open(step3_path+filename+".txt", 'w') as step3_cleaned:
            step3_cleaned.write('\n'.join([json.dumps(elt) for elt in new_file]))
        if not verbose:
            bar.update(i+1)
    if not verbose:
        bar.finish()


def integrate_triples(step1_path, step3_path, final_path, verbose):
    """
    Integrate step 1 and step 3 to create the final triples.
    :param step1_path: Path leading to step 1 triples
    :param step3_path: Path leading to step 2 triples
    :param final_path: Path where to write the final triples
    :param verbose: Debug
    :return: None
    """
    # Consider files for which we have both step 1 and step 3
    files_step_1 = [f.split("/")[-1].split(".")[0] for f in glob.glob(step1_path+"*.txt")]
    files_step_3 = [f.split("/")[-1].split(".")[0] for f in glob.glob(step3_path+"*.txt")]
    files = set(files_step_1).intersection(set(files_step_3))
    if not verbose:
        bar = progressbar.ProgressBar(maxval=len(files),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

    for i, filename in enumerate(files):
        if verbose:
            print('\n')
            print(filename)

        triples = []
        # Create a set of entity name to quickly check if a triple subject belongs in this set
        set_of_entities = set()
        # Link entity name to its type and link
        entities = {}
        with open(step1_path+filename+".txt", 'r') as step1_file:
            # Create a set of entities recognized by DBPedia
            for line in step1_file.readlines():
                l = json.loads(line.strip())
                entity_name = l[0]
                entity_link = l[1].split("/")[-1]
                entity_type = l[2]
                set_of_entities.add(l[0])
                entities[entity_name] = (entity_link, entity_type)

        with open(step3_path+filename+'.txt', 'r') as step3_file:
            for line in step3_file.readlines():
                # Parse each triple
                l = json.loads(line.strip())
                subject = l[0][0]
                relation = l[1][0]
                object = l[2][0]
                subject_is_entity = False
                object_is_entity = False

                subject_entity_name = ""
                subject_without_entity = ""
                object_entity_name = ""
                object_without_entity = ""
                for token in subject.split(" "):
                    # For each token, test if token in list of entities
                    subject_is_entity = subject_is_entity or (token in set_of_entities)
                    if token in set_of_entities:
                        subject_entity_name = token
                        l = subject.split(" ")
                        l.remove(subject_entity_name)
                        subject_without_entity = " ".join(l)
                for token in object.split(" "):
                    object_is_entity = object_is_entity or (token in set_of_entities)
                    if token in set_of_entities:
                        object_entity_name = token
                        l = object.split(" ")
                        l.remove(object_entity_name)
                        object_without_entity = " ".join(l)
                if object_is_entity and subject_is_entity:
                    triples.append(((subject_entity_name, subject_without_entity, entities[subject_entity_name]),
                                    relation,
                                    (object_entity_name, object_without_entity, entities[object_entity_name])))
        with open(final_path+filename+'.txt', 'w') as final_file:
            final_file.write('\n'.join([json.dumps(elt) for elt in triples]))
        if not verbose:
            bar.update(i+1)
    if not verbose:
        bar.finish()


def triples_integration(triples_path):
    step1_path = triples_path + "step1/"
    step2_path = triples_path + "step2/"
    step3_path = triples_path + "step3/"
    final_path = triples_path + "final/"
    if not os.path.isdir(final_path):
        os.mkdir(final_path)

    print("Disambiguate step 3 triples...")
    disambiguate(step2_path, step3_path, verbose=False)
    print("Remove identical step 3 triples...")
    remove_identical_triples(step3_path, verbose=False)
    # clear_duplicates(step3_path)
    print("Combine step 1 and step 3 to get final triples...")
    integrate_triples(step1_path, step3_path, final_path, verbose=False)
    print("Upload to Google Cloud...")
    upload_to_cloud_storage(triples_path + 'final/*.txt', 'final_triples/')


if __name__ == "__main__":
    step1_path = Config.TRIPLES + "step1/"
    step2_path = Config.TRIPLES + "step2/"
    step3_path = Config.TRIPLES + "step3/"
    final_path = Config.TRIPLES + "final/"
    if not os.path.isdir(final_path):
        os.mkdir(final_path)

    print("Disambiguate step 3 triples...")
    disambiguate(step2_path, step3_path, verbose=False)
    print("Remove identical step 3 triples...")
    remove_identical_triples(step3_path, verbose=False)
    # clear_duplicates(step3_path)
    print("Combine step 1 and step 3 to get final triples...")
    integrate_triples(step1_path, step3_path, final_path, verbose=False)
    print("Upload to Google Cloud...")
    upload_to_cloud_storage(Config.TRIPLES + 'final/*.txt', 'final_triples/')
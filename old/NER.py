from nltk.tag import StanfordNERTagger
from collections import Counter
from config import Config
import glob


def get_tag_from_file(file_path):
    with open(file_path, 'r') as file:
        text = "".join(file.readlines())
    text = text.split(" ")

    st = StanfordNERTagger(Config.PATH_TO_CLASSIFIER, Config.JAR, encoding="utf-8")
    tagged_words = st.tag(text)
    print(tagged_words)
    tagged_elements = [t for t in tagged_words if t[1] != 'O']
    print(tagged_elements)
    list_of_entities = set([t[1] for t in tagged_elements])

    entities = {entity: Counter() for entity in list_of_entities}
    for tag in entities.keys():
        entities[tag].update([t[0] for t in tagged_elements if t[1] == tag])

    return entities


def combine_dicts(a, b):
    merged_dict = dict()
    for k in set(b) & set(a):
        merged_dict[k] = a[k] + b[k]
    for k in set(b) - set(a):
        merged_dict[k] = b[k]
    for k in set(a) - set(b):
        merged_dict[k] = a[k]
    return merged_dict


def get_tag_from_dir(dir_path, limit=float('inf')):
    entities = dict()
    for (i,file_path) in enumerate(glob.glob(dir_path)):
        new_entities = get_tag_from_file(file_path)
        entities = combine_dicts(entities, new_entities)
        if i >= limit:
            break
    return entities


def pprint(entities):
    output = []
    for tag in entities.keys():
        output.append(tag)
        output.append(str(entities[tag].most_common(5)))
    return "\n".join(output)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    # parser.add_argument("-l", "--limit", type=int, default=10,  help="Maximum number of files to parse")
    #
    # A = parser.parse_args()
    #
    # entities = get_tag_from_dir(Config.TEXT_DATA_PATH, limit=A.limit)
    # output = pprint(entities)
    #
    # with open("entities.txt", "w") as output_file:
    #     output_file.write(output)

    get_tag_from_file("output/61051533913362_Phosphates Weekly Report 01-19-2017.txt")


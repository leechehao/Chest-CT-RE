from typing import Optional

import os
import argparse
import random
import json

import numpy as np


class TextInfo:
    def __init__(self, text: str, point: int = 0):
        self.text = text
        self.starts = []
        self.ends = []
        self.ents = []
        self.rels = []
        self.tags: Optional[list[str]] = None
        for word in text.split():
            end = point + len(word)
            self.starts.append(point)
            self.ends.append(end)
            point = end + 1  # 所有 word 之間都以一個空白隔開

    def create_tags(self):
        self.tags = ["O"] * len(self.starts)


def insert_string_at_index(original_string, insert_string, index):
    """
    在指定索引處向字符串中插入另一個字符串。

    Args:
        original_string (_type_): 原始字符串。
        insert_string (_type_): 要插入的字符串。
        index (_type_): 插入位置的索引。

    Returns:
        _type_: 修改後的新字符串。
    """
    return original_string[:index] + insert_string + original_string[index:]


class RelationExample:
    def __init__(self, text: str, root_id: str) -> None:
        self.text = text
        self.root_id = root_id
        self.ents = []

    def process(self):
        self.ents = sorted(self.ents, key=lambda x: x[1])
        is_find_root = False
        for ent in self.ents:
            if ent[4] == self.root_id:
                is_find_root = True
                self.text = insert_string_at_index(self.text, "$ ", ent[1])
                ent[1] += 2
                ent[2] += 2
                self.text = insert_string_at_index(self.text, " $", ent[2])
                continue
            if is_find_root:
                ent[1] += 4
                ent[2] += 4


def process_annotation(annotation: dict) -> tuple:
    text = annotation["value"]["text"]
    leading_whitespace_length = len(text) - len(text.lstrip())
    start = annotation["value"]["start"] + leading_whitespace_length
    end = start + len(text.strip())
    return text, start, end, annotation["value"]["labels"][0], annotation["id"]


def extract_annotation_info(json_data: list[dict], field_name: str) -> list[TextInfo]:
    label_data = []
    texts = []
    for example in json_data:
        if example["data"][field_name] in texts:
            continue
        example_info = TextInfo(text=example["data"][field_name])
        texts.append(example["data"][field_name])
        for annotation in example["annotations"][0]["result"]:
            if "value" in annotation:  # 如果有 "value" 的 key，代表為 entity 的標註，反之為 relation 的標註
                example_info.ents.append(process_annotation(annotation))
            else:
                example_info.rels.append((annotation["from_id"], annotation["to_id"]))
        label_data.append(example_info)
    return label_data


def segment_ligatures_by_ent(label_data: list[TextInfo]) -> None:
    for label in label_data:
        for ent in label.ents:
            ent_info = TextInfo(text=ent[0], point=ent[1])
            for i in range(len(ent_info.starts)):  # entity 中的 word index(i)
                for j, (start, end) in enumerate(zip(label.starts, label.ends)):  # example 中的 word index(j)
                    if start == ent_info.starts[i] and ent_info.ends[i] < end:
                        # example   -> Massat lung . [(0, 6), (7, 11), (12, 13)]
                        # ent       -> Mass (0, 4)
                        # condition -> 0 == 0 and 4 < 6
                        # result    -> (0, 6) => (0, 4), (4, 6)
                        label.starts[j] = ent_info.ends[i]
                        label.ends[j] = end
                        label.starts.insert(j, start)
                        label.ends.insert(j, ent_info.ends[i])
                        break
                    elif start < ent_info.starts[i] and ent_info.ends[i] == end:
                        # example   -> Mass atlung . [(0, 4), (5, 11), (12, 13)]
                        # ent       -> lung (7, 11)
                        # condition -> 5 < 7 and 11 == 11
                        # result    -> (5, 11) => (5, 7), (7, 11)
                        label.starts[j] = ent_info.starts[i]
                        label.ends[j] = end
                        label.starts.insert(j, start)
                        label.ends.insert(j, ent_info.starts[i])
                        break
                    elif start < ent_info.starts[i] and ent_info.ends[i] < end:
                        # example   -> Mass atlung. [(0, 4), (5, 12)]
                        # ent       -> lung (7, 11)
                        # condition -> 5 < 7 and 11 < 12
                        # result    -> (5, 12) => (5, 7), (7, 11), (11, 12)
                        label.starts[j] = ent_info.ends[i]
                        label.ends[j] = end
                        label.starts.insert(j, ent_info.starts[i])
                        label.ends.insert(j, ent_info.ends[i])
                        label.starts.insert(j, start)
                        label.ends.insert(j, ent_info.starts[i])


def convert_relation_to_ner_data(label_data: list[TextInfo]) -> list[TextInfo]:
    results = []
    for label in label_data:
        ent_map = {ent[4]: list(ent) for ent in label.ents}
        data = {}
        for relation in label.rels:
            if relation[0] not in data:
                data[relation[0]] = RelationExample(text=label.text, root_id=relation[0])
                data[relation[0]].ents.append(ent_map[relation[0]])
            data[relation[0]].ents.append(ent_map[relation[1]])
        for key, rel_example in data.items():
            rel_example.process()
            example = TextInfo(text=rel_example.text)
            example.ents = rel_example.ents
            results.append(example)
    return results


def tag_BIO(label_data: list[TextInfo]) -> None:
    for label in label_data:
        label.create_tags()
        for ent in label.ents:
            ent_info = TextInfo(text=ent[0], point=ent[1])
            for i in range(len(ent_info.starts)):
                for j, (start, end) in enumerate(zip(label.starts, label.ends)):
                    if start == ent_info.starts[i] and ent_info.ends[i] == end:
                        label.tags[j] = f"B-{ent[3]}" if i == 0 else f"I-{ent[3]}"


def train_test_split(*arrays, test_size: float = 0.25):
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    # 確保所有數組的長度相同
    length = len(arrays[0])
    for array in arrays:
        if len(array) != length:
            raise ValueError("All arrays must have the same length")

    # 生成隨機排列的索引
    indices = list(range(length))
    random.shuffle(indices)

    # 計算分割點
    split_idx = int(length * (1 - test_size))

    # 分割每個數組，並返回結果
    result = []
    for array in arrays:
        result.extend([np.array(array)[indices[:split_idx]].tolist(), np.array(array)[indices[split_idx:]].tolist()])
    return result


def write_content(split_data: list[TextInfo]) -> str | set[str]:
    file_content = ""
    tag_set = set()
    for label in split_data:
        for start, end, tag in zip(label.starts, label.ends, label.tags):
            tag_set.add(tag.split("-")[-1])
            file_content += f"{label.text[start:end]} {tag}\n"
        file_content += "\n"
    return file_content, tag_set


def write_file(file_path: str, content: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="This is annotated data for a NER task, coming from Label Studio in JSON format.")
    parser.add_argument("--field_name", type=str, required=True, help="The field name of the example in input_file.")
    parser.add_argument("--output_dir", type=str, required=True, help="The dataset for a NER task in CONLL format.")
    parser.add_argument("--seed", default=1314, type=int, help="Random seed.")
    args = parser.parse_args()

    with open(args.input_file) as f:
        json_data = json.load(f)

        label_data = extract_annotation_info(json_data, args.field_name)
        segment_ligatures_by_ent(label_data)

    relation_data = convert_relation_to_ner_data(label_data)
    tag_BIO(relation_data)

    train_data, test_data = train_test_split(relation_data, test_size=0.2)
    train_data, validation_data = train_test_split(train_data, test_size=0.125)

    if os.path.exists(args.output_dir):
        raise FileExistsError(f"The folder '{args.output_dir}' already exists!")

    train_content, train_tag_set = write_content(train_data)
    validation_content, validation_tag_set = write_content(validation_data)
    test_content, test_tag_set = write_content(test_data)

    if validation_tag_set > train_tag_set or test_tag_set > train_tag_set:
        raise ValueError(f"The labels in the training data do not include those found in the validation and test data.")

    data_dir = f"{args.output_dir}/data"
    os.makedirs(data_dir)
    write_file(f"{data_dir}/train.conll", train_content)
    write_file(f"{data_dir}/validation.conll", validation_content)
    write_file(f"{data_dir}/test.conll", test_content)


if __name__ == '__main__':
    main()

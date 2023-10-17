from argparse import ArgumentParser
import json

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--context_dir", 
                        default='ntuadl2023hw1/context.json',
                        type=str)
    parser.add_argument("--data_dir",
                        default='ntuadl2023hw1/train.json',
                        type=str)
    parser.add_argument("--output_dir",
                        default='ntuadl2023hw1/train_qa.json',
                        type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    with open(args.data_dir, 'r') as json_file:
        data_list = json.load(json_file)
    with open(args.context_dir, 'r') as json_file:
        context_list = json.load(json_file)

    ans = []
    for data in data_list:
        data["answers"] = {"answer_start":[data["answer"]["start"]], "text":[data["answer"]["text"]]}
        data["context"] = context_list[data["relevant"]]
        data.pop("paragraphs")
        data.pop("relevant")
        data.pop("answer")
        ans.append(data)

    data_json = {"data": ans}
    json.dump(data_json, open(args.output_dir, 'w'), indent=2, ensure_ascii=False)
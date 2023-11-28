import argparse
import os.path
import os ,sys
sys.path.append("..")
from lens import Lens, LensProcessor
from PIL import Image
import pandas as pd
import torch
from tqdm import tqdm
from utils.universal_utils import get_dataset_classes

parser =  argparse.ArgumentParser()
parser.add_argument('-s','--source_list')
parser.add_argument('-t','--target_list')
parser.add_argument('--dataset_name',default='OfficeHome')
parser.add_argument('--save_path', type=str)
parser.add_argument('--base_path')
args = parser.parse_args()

# parse data_list_file
def parse_data_file(file_name: str):
    """Parse file to data list

    Args:
        file_name (str): The path of data file
        return (list): List of (image path, class_index) tuples
    """
    with open(file_name, "r") as f:
        data_list = []
        for line in f.readlines():
            split_line = line.split()
            target = split_line[-1]
            path = ''.join(split_line[:-1])
            if not os.path.isabs(path):
                path = os.path.join(args.base_path, path)
            target = int(target)
            data_list.append((path, target))
    return data_list

CLASSES = get_dataset_classes(args.dataset_name)

def main():
    source_list_file = os.path.join(os.getcwd(),args.source_list)
    #target_list_file = os.path.join(os.getcwd(), args.target_list)

    source_list = parse_data_file(source_list_file)
    #target_list = parse_data_file(target_list_file)

    save_file = os.path.join(args.save_path)
    # df = pd.DataFrame(columns=['descriptions','categories'])
    output_data = {'descriptions':[], 'categories':[]}

    lens = Lens()
    processor = LensProcessor()

    for source_item in tqdm(source_list):
        img_url = source_item[0]
        raw_image = Image.open(img_url).convert('RGB')
        with torch.no_grad():
            samples = processor([raw_image], ['Tell me about the image?'])
            lens(samples)
        output_data['descriptions'].append(samples["prompts"][0])
        output_data['categories'].append(CLASSES[source_item[1]])

    df = pd.DataFrame(output_data)
    df.to_csv(save_file)

if __name__ == '__main__':
    main()
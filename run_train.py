from vf_model import vf_model
import argparse
import os

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='run eval')
    # parser.add_argument('--save_folder', '-i',
    #                     help='folder with input files')
    # parser.add_argument('--model', '-m',
    #                     help='weights of the enhancement model in .h5 format')
    # args = parser.parse_args()

    # # use the GPU with idx 0
    # os.environ["CUDA_VISIBLE_DEVICES"]='0'
    # # activate this for some reproducibility
    # os.environ['TF_DETERMINISTIC_OPS'] ='1'
    Name = "test"
    csvPath = "./dataset/cv-valid-train-simplified.csv"
    dataset_path = "./dataset/cv-valid-train/"

    model = vf_model(name=Name, csvPath=csvPath, dataset_path=dataset_path)
    model.build_vf_model()
    model.compile_model()

    model.train_model()
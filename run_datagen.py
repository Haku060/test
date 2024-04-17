from vf_model import vf_model


if __name__ == '__main__':

    Name = "test"
    csvPath = "./dataset/cv-valid-train-simplified.csv"
    dataset_path = "./dataset/cv-valid-train/"
    
    model = vf_model(name=Name, csvPath=csvPath, dataset_path=dataset_path)
    model.dataset_generator()
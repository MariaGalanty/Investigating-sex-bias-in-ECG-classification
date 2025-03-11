# Imports
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from data.data_loader import *
from models.resnet import *
from models.cinc_isibrno import *
from utils.train import *
from utils.eval import *
from utils.focal_loss import *
from utils.helpers import *
import argparse
import json
import os
import torch
import numpy as np


# Arguments to be passed from the command line
def parse_args():
    parser = argparse.ArgumentParser(description='Provide necessary arguments for training the model')

    parser.add_argument('-r', '--sex_ratio', type=str, default='1_0',
                        help="The sex ratio in the training data males vs females, options: [1_0, 0.75_0.25, 0.50_0.50, 0.25_0.75, 0_1]")
    parser.add_argument('-f', '--fold', type=int, default=0,
                        help="fold to be use as a test set(0-4)")
    parser.add_argument('-m', '--model', type=str, default='cinc',
                        help="Classifier algorithm, choose from [cinc, resnet34, resnet50, resnet101]")
    parser.add_argument('-exp', '--experiment_id', type=str, default='resnet101',
                        help="Experiment ID")
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('-w', '--weights', '--weights', type=int, default=0,
                        help="Oversampling minority classes")
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help="Number of epochs")
    parser.add_argument('-l', '--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('-loss', '--loss', type=str, default='focal',
                        help="Loss function")
    parser.add_argument('-p', '--patience', type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument('-n', '--normalize', type=str, default='z-score',
                        help="Normalize the data: ['z-score', 'min_max', None]")
    parser.add_argument('-t', '--length', type=int, default=4096,
                        help="Length of the signal")
    parser.add_argument('-c', '--class_nr', type=int, default=4,
                        help="Number of classes")
    parser.add_argument('-g', '--gpu', type=int, nargs='*', default=[2,3,4], #[5, 6],
                        help="GPU number(s)")
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help="GPU number")

    return vars(parser.parse_args())


def save_args(args, results_dir):
    with open(f'{results_dir}/args.json', 'w') as f:
        json.dump(args, f, indent=4)
def setup_directories(experiment_id):
    results_dir = f'results/{experiment_id}'
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
def setup_model(args):
    if args["model"] == 'cinc':
        model = ModelCinc(args["class_nr"])
    elif args["model"] == 'resnet34':
        model = ResNetModel(name='resnet34', head='mlp', feat_dim=args["class_nr"])
    elif args["model"] == 'resnet50':
        model = ResNetModel(name='resnet50', head='mlp', feat_dim=args["class_nr"])
    elif args["model"] == 'resnet101':
        model = ResNetModel(name='resnet101', head='mlp', feat_dim=args["class_nr"])
    else:
        print('Model not defined - use \'cinc\' or \'resnet\'')
        sys.exit()

    #model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=1e-5)
    if args["loss"] == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args["loss"] == 'focal':
        criterion = FocalLoss(alpha = 0.25, gamma=2.0)
        print(criterion)
    return model, optimizer, criterion


def main():
    args = parse_args()
    print(args)
    set_seed(args["seed"])
    results_dir = setup_directories(args["experiment_id"])
    save_args(args, results_dir)

    data_directory = ['/home/maria/data/PhysioNet2021_preprocessed/WFDB_PTBXL',
                      '/home/maria/data/PhysioNet2021_preprocessed/WFDB_CPSC2018',
                      '/home/maria/data/PhysioNet2021_preprocessed/WFDB_CPSC2018_2',
                      '/home/maria/data/PhysioNet2021_preprocessed/WFDB_Ga',
                      '/home/maria/data/PhysioNet2021_preprocessed/WFDB_ChapmanShaoxing',
                      '/home/maria/data/PhysioNet2021_preprocessed/WFDB_Ningbo']

    print('Finding header and recording files...')
    print(data_directory)
    header_files, recording_files = find_challenge_files(data_directory)
    cinc_dataset = dataset(header_files, nr_leads=12, length=args["length"], normalize=args["normalize"], equivalent_cl='sinus_mi_hyp', return_source=False)
    classes_info(cinc_dataset)

    # getting the dataset division
    # Path to the JSON file
    file_path = os.path.join("/home/maria/projects/sex_bias_hr_detection/data/dataset_division.json")
    fold = args['fold']

    # Open and read the JSON file
    with open(file_path, "r") as f:
        dataset_division = json.load(f)[str(fold)]
    male_test_idx = dataset_division["male_balanced_test_idx"]
    female_test_idx = dataset_division["female_balanced_test_idx"]

    if args["sex_ratio"] == '1_0':
        train_idx = dataset_division["train_idx_100_0"]
        val_idx = dataset_division["train_idx_100_0"]
    elif args["sex_ratio"] == '0.75_0.25':
        train_idx = dataset_division["train_idx_75_25"]
        val_idx = dataset_division["train_idx_75_25"]
    elif args["sex_ratio"] == '0.50_0.50':
        train_idx = dataset_division["train_idx_50_50"]
        val_idx = dataset_division["train_idx_50_50"]
    elif args["sex_ratio"] == '0.25_0.75':
        train_idx = dataset_division["train_idx_25_75"]
        val_idx = dataset_division["train_idx_25_75"]
    elif args["sex_ratio"] == '0_1':
        train_idx = dataset_division["train_idx_0_100"]
        val_idx = dataset_division["train_idx_0_100"]

    train = Subset(cinc_dataset, train_idx)
    val = Subset(cinc_dataset, val_idx)
    male_test = Subset(cinc_dataset, male_test_idx)
    female_test = Subset(cinc_dataset, female_test_idx)

    if args["weights"] == 1:
        print("weigths")
        # Calculate class distribution (frequency of each class)
        train_labels = [train.dataset[i][1] for i in train.indices]
        all_labels = np.array(train_labels)

        # Step 1: Exclude samples with all zeros (no class assigned)
        non_zero_samples = all_labels.sum(axis=1) > 0  # Only keep samples that have at least one '1' (label present)
        filtered_labels = all_labels[non_zero_samples]  # Filter out samples with all zeros

        # Step 2: Calculate the frequency of each class (considering each label separately)
        class_counts = np.sum(filtered_labels == 1, axis=0)  # Count occurrences of 1 per class across all samples
        class_weights = 1. / class_counts  # Inverse of frequency for oversampling

        sample_weights = []

        for sample in all_labels:
            sample_weight = sum(class_weights[i] for i, label in enumerate(sample) if label == 1)
            sample_weights.append(sample_weight)

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        SHUFFLE = False
    else:
        print("no weigths")
        SHUFFLE = True
        sampler = None

    train_loader = DataLoader(dataset=train, batch_size=args["batch_size"], shuffle=SHUFFLE, collate_fn=collate,
                              num_workers=0, pin_memory=True, drop_last=True, sampler=sampler)
    val_loader = DataLoader(dataset=val, batch_size=args["batch_size"], shuffle=True, collate_fn=collate,
                            num_workers=0, pin_memory=True, drop_last=False)
    male_test_loader = DataLoader(dataset=male_test, batch_size=args["batch_size"], shuffle=False, collate_fn=collate,
                             num_workers=0, pin_memory=True, drop_last=False)
    female_test_loader = DataLoader(dataset=female_test, batch_size=args["batch_size"], shuffle=False, collate_fn=collate,
                                  num_workers=0, pin_memory=True, drop_last=False)

    model, optimizer, criterion = setup_model(args)

    #multiple gpus if needed
    print(type(args['gpu']))
    if type(args['gpu']) == int:
        DEVICE = torch.device(f"cuda:{args['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    else:
        if len(args['gpu']) == 1:
            DEVICE = torch.device(f"cuda:{args['gpu'][0]}") if torch.cuda.is_available() else torch.device("cpu")
        else:
            DEVICE = torch.device(f"cuda:{args['gpu'][0]}") if torch.cuda.is_available() else torch.device("cpu")
            model = nn.DataParallel(model, device_ids=args['gpu'])
    print(DEVICE)

    #sys.exit() #, DEVICE)
    print(args)

    train_loop(model, train_loader, val_loader, args["epochs"], args["patience"], optimizer, criterion,
               DEVICE, args["class_nr"], args["experiment_id"])

    labels, outputs = get_outputs(model, val_loader, device=DEVICE)
    thresholds= thresholds_roc_auc(labels, outputs)
    report_roc_f1_plot(model, male_test_loader, thresholds, thresholds_id='roc', device=DEVICE, class_nr=args["class_nr"],
                       experiment_id=args["experiment_id"], dataset_name='male_test_set', display=False)
    report_roc_f1_plot(model, female_test_loader, thresholds, thresholds_id='roc', device=DEVICE, class_nr=args["class_nr"],
                       experiment_id=args["experiment_id"], dataset_name='female_test_set', display=False)



    print("precision")
    thresholds = thresholds_balance_precision_recall(labels, outputs)
    report_roc_f1_plot(model, male_test_loader, thresholds, thresholds_id='precision_recall', device=DEVICE,
                       class_nr=args["class_nr"], experiment_id=args["experiment_id"], dataset_name='male_test_set',
                       display=False)
    report_roc_f1_plot(model, female_test_loader, thresholds, thresholds_id='precision_recall', device=DEVICE,
                       class_nr=args["class_nr"],  experiment_id=args["experiment_id"], dataset_name='female_test_set',
                       display=False)

    print("youden")
    thresholds = thresholds_youden(labels, outputs)
    report_roc_f1_plot(model, male_test_loader, thresholds, thresholds_id='youden', device=DEVICE,
                       class_nr=args["class_nr"], experiment_id=args["experiment_id"], dataset_name='male_test_set', display=False)
    report_roc_f1_plot(model, female_test_loader, thresholds, thresholds_id='youden', device=DEVICE,
                       class_nr=args["class_nr"], experiment_id=args["experiment_id"], dataset_name='female_test_set',
                       display=False)


if __name__ == "__main__":
    main()
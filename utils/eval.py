import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, confusion_matrix, roc_curve, recall_score, precision_score, accuracy_score
import math
import torch
import numpy as np
import os

def get_outputs(model, eval_generator, device):
    labels = []
    outputs = []

    for test_local_batch, test_local_labels, lead_idx in eval_generator:
        test_local_labels = test_local_labels.float().to(device)
        lead_idx = lead_idx.float().to(device)
        model.eval()
        # Run the forward pass
        with torch.no_grad():
            if 'ModelCinc' in model.__class__.__name__:
                test_local_batch = test_local_batch.unsqueeze(2).float().to(device)
                test_outputs = model(test_local_batch, lead_idx).float()
            else:
                test_local_batch = test_local_batch.float().to(device)
                test_outputs = model(test_local_batch).float()

        labels.append(test_local_labels.cpu().detach().numpy().tolist())
        outputs.append(test_outputs.cpu().detach().numpy().tolist())

    labels = [val for sublist in labels for val in sublist]
    outputs = [val for sublist in outputs for val in sublist]
    labels = np.array(labels)
    outputs = np.array(outputs)
    return labels, outputs

def thresholds_youden(labels, outputs):
    thresholds = []
    for i in range(labels.shape[1]):
        fpr, tpr, th = roc_curve(labels[:, i], outputs[:, i])
        optimal_idx = np.argmax(tpr - fpr)
        thresholds.append(th[optimal_idx])
    return thresholds

def thresholds_balance_precision_recall(labels, outputs):
    thresholds = []
    for i in range(labels.shape[1]):
        precision, recall, th = precision_recall_curve(labels[:, i], outputs[:, i])
        # Compute F1 scores, avoiding division by zero
        denominator = precision + recall
        f1_scores = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
        optimal_idx = np.argmax(f1_scores)
        thresholds.append(th[optimal_idx])
    return thresholds

def thresholds_roc_auc(labels, outputs):
    thresholds = []
    for i in range(labels.shape[1]):
        fpr, tpr, th = roc_curve(labels[:, i], outputs[:, i])
        k = np.arange(len(tpr))
        gmeans = [math.sqrt(tpr[k] * (1 - fpr[k])) for k in range(len(tpr))]
        ix = max(enumerate(gmeans), key=lambda x: x[1])[0]
        thresholds.append(th[ix])

    return thresholds


def report_roc_f1_plot(model, test_generator, thresholds, thresholds_id = 'roc', device=None, class_nr =6,
                       experiment_id="test", dataset_name = "no name dataset", display=False):
    labels = []
    outputs = []
    test_roc = []

    for test_local_batch, test_local_labels, lead_idx in test_generator:
        test_local_labels = test_local_labels.float().to(device)
        lead_idx = lead_idx.float().to(device)
        model.eval()
        # Run the forward pass
        with torch.no_grad():
            if 'ModelCinc' in model.__class__.__name__:
                test_local_batch = test_local_batch.unsqueeze(2).float().to(device)
                test_outputs = model(test_local_batch, lead_idx).float()
            else:
                test_local_batch = test_local_batch.float().to(device)
                test_outputs = model(test_local_batch).float()
        labels.append(test_local_labels.cpu().detach().numpy().tolist())
        outputs.append(test_outputs.cpu().detach().numpy().tolist())

    labels = [val for sublist in labels for val in sublist]
    outputs = [val for sublist in outputs for val in sublist]
    labels = np.array(labels)
    outputs = np.array(outputs)
    results_dir = f"results/{experiment_id}"
    os.makedirs(results_dir, exist_ok=True)

    if class_nr == 6:
        class_names = ['Atrial Fibrillation', 'Atrial Flutter', 'Sinus Rhythm', 'Pacing Rhythm', 'Sinus Bradycardia',
                       'Sinus Tachycardia']
        predicted_classes = [
            [1 if (x > thresholds[0]) else 0 for x in outputs[:, 0]],
            [1 if (x > thresholds[1]) else 0 for x in outputs[:, 1]],
            [1 if (x > thresholds[2]) else 0 for x in outputs[:, 2]],
            [1 if (x > thresholds[3]) else 0 for x in outputs[:, 3]],
            [1 if (x > thresholds[4]) else 0 for x in outputs[:, 4]],
            [1 if (x > thresholds[5]) else 0 for x in outputs[:, 5]]
        ]
    elif class_nr == 2:
        class_names = ['Atrial Fibrillation', 'Sinus Rhythm']
        predicted_classes = [
            [1 if (x > thresholds[0]) else 0 for x in outputs[:, 0]],
            [1 if (x > thresholds[1]) else 0 for x in outputs[:, 1]]
        ]
    elif class_nr == 4:
        class_names = ['Atrial Fibrillation', 'Sinus Rhythm', 'Myocardial infarction', 'Hypertrophy']
        predicted_classes = [
            [1 if (x > thresholds[0]) else 0 for x in outputs[:, 0]],
            [1 if (x > thresholds[1]) else 0 for x in outputs[:, 1]],
            [1 if (x > thresholds[2]) else 0 for x in outputs[:, 2]],
            [1 if (x > thresholds[3]) else 0 for x in outputs[:, 3]]
        ]

    # Set up plotting area
    plt.figure(0).clf()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print('Dataset name: ', dataset_name)
    # Save results to file
    with open(os.path.join(results_dir, f"{dataset_name}_results.txt"), "a") as f: #{experiment_id}_{dataset_name}_{thresholds_id}_
        f.write(f"Thresholds: {thresholds_id}\n")
        for i in range(len(class_names)):
            print(class_names[i])
            if len(np.unique(labels[:, i])) > 1:
                # Roc
                roc_auc = roc_auc_score(labels[:, i], outputs[:, i])
                print('ROC AUC: ', roc_auc)
                # Confusion matrix
                cm = confusion_matrix(labels[:, i], predicted_classes[i])
                #tn, fp, fn, tp = cm.ravel()
                print('Confusion matrix: ')
                print(cm)
                # Recall
                recall = recall_score(labels[:, i], predicted_classes[i], zero_division=0)
                print('Recall: ', recall)
                # Precision
                precision = precision_score(labels[:, i], predicted_classes[i], zero_division=0)
                print('Precision: ', precision)
                # Accuracy
                accuracy = accuracy_score(labels[:, i], predicted_classes[i])
                print('Accuracy: ', accuracy)
                # f1 score
                f1 = f1_score(labels[:, i], predicted_classes[i])
                print('F1 score: ', f1)
                print('--------------------------------------')
                # Save results to file
                f.write(f"Class: {class_names[i]}\n")
                f.write(f"ROC AUC: {roc_auc}\n")
                f.write(f"Confusion matrix:\n{cm}\n")
                f.write(f"Recall: {recall}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 score: {f1}\n")
                f.write('--------------------------------------\n')

                # Add for plotting
                fpr, tpr, th = roc_curve(labels[:, i], outputs[:, i])
                plt.plot(fpr, tpr, label=class_names[i])
            else:
                print('Not possible to assess')
                f.write(f"Class: {class_names[i]}\n")
                f.write('Not possible to assess\n')
                f.write('--------------------------------------\n')
    if thresholds_id == 'roc':
        plt.legend()
        if display:
            plt.show()
        else:
            plt.savefig(os.path.join(results_dir, f"{dataset_name}_roc_curve.png")) #{experiment_id}_

def indexes(classes, true_labels, predicted_classes):

# Initialize dictionaries to store indexes
    indexes = {
        classes[0]: {'TP': [], 'FP': [], 'TN': [], 'FN': []},
        classes[1]: {'TP': [], 'FP': [], 'TN': [], 'FN': []}
    }

    # Iterate over each class
    for class_idx in range(true_labels.shape[1]):
        for i in range(len(true_labels)):
            if true_labels[i, class_idx] == 1 and predicted_classes[i, class_idx] == 1:
                indexes[classes[class_idx]]['TP'].append(i)
            elif true_labels[i, class_idx] == 0 and predicted_classes[i, class_idx] == 1:
                indexes[classes[class_idx]]['FP'].append(i)
            elif true_labels[i, class_idx] == 0 and predicted_classes[i, class_idx] == 0:
                indexes[classes[class_idx]]['TN'].append(i)
            elif true_labels[i, class_idx] == 1 and predicted_classes[i, class_idx] == 0:
                indexes[classes[class_idx]]['FN'].append(i)
    return indexes

def report_index(model, test_generator, thresholds, thresholds_id = 'roc', device=None, class_nr =6,
                       experiment_id="test", dataset_name = None, save=True):
    labels = []
    outputs = []
    test_roc = []

    for test_local_batch, test_local_labels, lead_idx in test_generator:
        test_local_labels = test_local_labels.float().to(device)
        lead_idx = lead_idx.float().to(device)
        model.eval()
        # Run the forward pass
        with torch.no_grad():
            if 'ModelCinc' in model.__class__.__name__:
                test_local_batch = test_local_batch.unsqueeze(2).float().to(device)
                test_outputs = model(test_local_batch, lead_idx).float()
            else:
                test_local_batch = test_local_batch.float().to(device)
                test_outputs = model(test_local_batch).float()
        labels.append(test_local_labels.cpu().detach().numpy().tolist())
        outputs.append(test_outputs.cpu().detach().numpy().tolist())

    labels = [val for sublist in labels for val in sublist]
    outputs = [val for sublist in outputs for val in sublist]
    labels = np.array(labels)
    outputs = np.array(outputs)
    if class_nr == 2:
        class_names = ['Atrial Fibrillation', 'Sinus Rhythm']
        predicted_classes = [
            [1 if (x > thresholds[0]) else 0 for x in outputs[:, 0]],
            [1 if (x > thresholds[1]) else 0 for x in outputs[:, 1]]
        ]
        #print(predicted_classes)

    # Set up plotting area
    plt.figure(0).clf()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print('Dataset name: ', dataset_name)
    # Save results to file
    #with open(os.path.join(results_dir, f"{experiment_id}_{dataset_name}_{thresholds_id}_results.txt"), "w") as f:
    for i in range(len(class_names)):
        print(class_names[i])
        if len(np.unique(labels[:, i])) > 1:
            # Roc
            roc_auc = roc_auc_score(labels[:, i], outputs[:, i])
            print('ROC AUC: ', roc_auc)
            # Confusion matrix
            cm = confusion_matrix(labels[:, i], predicted_classes[i])
            print('Confusion matrix: ')
            print(cm)
            # Recall
            recall = recall_score(labels[:, i], predicted_classes[i], zero_division=0)
            print('Recall: ', recall)
            # Precision
            precision = precision_score(labels[:, i], predicted_classes[i], zero_division=0)
            print('Precision: ', precision)
            # Accuracy
            accuracy = accuracy_score(labels[:, i], predicted_classes[i])
            print('Accuracy: ', accuracy)
            # f1 score
            f1 = f1_score(labels[:, i], predicted_classes[i])
            print('F1 score: ', f1)
            print('--------------------------------------')
            # Add for plotting
            fpr, tpr, th = roc_curve(labels[:, i], outputs[:, i])
            plt.plot(fpr, tpr, label=class_names[i])
        else:
            print('Not possible to assess')
    plt.legend()
    plt.show()
    predicted_classes = [[predicted_classes[0][i], predicted_classes[1][i]] for i in range(len(predicted_classes[0]))]
    predicted_classes = np.array(predicted_classes)
    return labels, predicted_classes
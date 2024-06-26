import os
import random
import time
from typing import cast
import numpy as np
import matplotlib.pyplot as plt
import timm
import torch
import torchvision
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from fragment_splitter import CustomFragmentLoader
from model_train_logger import set_config_for_logger
from thyroid_dataset import ThyroidDataset
from thyroid_ml_model import ThyroidClassificationModel
from transformation import get_transformation


@torch.no_grad()
def validate(model, data_loader, loss_function=None, show_tqdm=False):
    class_set = sorted(data_loader.dataset.class_to_idx_dict.values())

    loss_values = []
    y_preds = []
    y_targets = []
    y_positive_scores = []

    for images, labels in (data_loader if not show_tqdm else tqdm(data_loader)):
        images = images.to(Config.available_device)
        labels = labels.to(Config.available_device)
        x = model(images, validate=True)
        if loss_function:
            loss_values.append(loss_function(x, labels))
        values, preds = torch.max(x, 1)

        y_positive_scores += x[:, 1].cpu()
        y_preds += preds.cpu()
        y_targets += labels.cpu()

    cf_matrix = confusion_matrix(y_targets, y_preds, normalize="true")

    class_accuracies = [cf_matrix[c][c] for c in class_set]
    acc = sum(class_accuracies)
    acc /= len(class_set)
    # TN|FN
    # FP|TP
    fpr, tpr, _ = roc_curve(y_targets, y_positive_scores)
    auc = roc_auc_score(y_targets, y_positive_scores)
    if loss_function:
        loss = sum(loss_values)
        loss /= len(loss_values)
        return acc * 100, cf_matrix, (fpr, tpr, auc), loss
    return acc * 100, cf_matrix, (fpr, tpr, auc)


def get_save_state_dirs(config_label, epoch=None):
    trains_state_dir = "./train_state"
    if not os.path.isdir(trains_state_dir):
        os.mkdir(trains_state_dir)
    config_train_dir = os.path.join(trains_state_dir, config_label)
    if not os.path.isdir(config_train_dir):
        os.mkdir(config_train_dir)
    if epoch is not None:
        save_state_dir = os.path.join(config_train_dir, f"epoch-{epoch}")
        if not os.path.isdir(save_state_dir):
            os.mkdir(save_state_dir)
    else:
        save_state_dir = None
    return trains_state_dir, config_train_dir, save_state_dir


def plot_and_save_model_per_epoch(epoch,
                                  model_to_save,
                                  val_acc_list,
                                  train_acc_list,
                                  val_loss_list,
                                  train_loss_list,
                                  config_label):
    trains_state_dir, config_train_dir, save_state_dir = get_save_state_dirs(config_label, epoch)

    fig_save_path = os.path.join(config_train_dir, "val_train_acc.jpeg")
    plt.plot(range(len(val_acc_list)), val_acc_list, label="validation")
    plt.plot(range(len(train_acc_list)), train_acc_list, label="train")
    plt.legend(loc="lower right")
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.savefig(fig_save_path)
    plt.clf()

    fig_save_path = os.path.join(config_train_dir, "val_train_loss.jpeg")
    plt.plot(range(len(val_loss_list)), val_loss_list, label="validation")
    plt.plot(range(len(train_loss_list)), train_loss_list, label="train")
    plt.legend(loc="lower right")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(fig_save_path)
    plt.clf()

    if model_to_save:
        model_save_path = os.path.join(save_state_dir, "model.state")
        model_to_save.save_model(model_save_path)


def save_auc_roc_chart_for_test(test_fpr, test_tpr, test_auc_score, config_label, epoch):
    trains_state_dir, config_train_dir, save_dir = get_save_state_dirs(config_label, epoch)
    fig_save_path = os.path.join(save_dir, f"test_roc_{time.time()}.jpeg")
    plt.plot(test_fpr, test_tpr, label="test, auc=" + str(test_auc_score))
    plt.legend(loc="lower right")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig(fig_save_path)
    plt.clf()


def calculate_test(image_model, epoch, test_data_loader, logger, config_name, show_tqdm=False):
    image_model.eval()
    test_acc, test_c_acc, (test_FPR, test_TPR, test_auc_score) = validate(image_model,
                                                                          test_data_loader,
                                                                          show_tqdm=show_tqdm)
    test_acc = float(test_acc)

    save_auc_roc_chart_for_test(test_FPR, test_TPR, test_auc_score, config_name, epoch)
    logger.info(f'Test|Epoch:{epoch}|Accuracy:{round(test_acc, 4)}, {test_c_acc}%')


def train_model(base_model, config_base_name, train_val_test_data_loaders, augmentation,
                adaptation_sample_dataset=None,
                train_model_flag=True,
                load_model_from_dir=None):
    config_name = f"{config_base_name}-{augmentation}-{','.join(Config.class_idx_dict.keys())}"

    logger = set_config_for_logger(config_name)
    logger.info(f"training config: {config_name}")
    try:
        _is_inception = type(base_model) == torchvision.models.inception.Inception3
        train_data_loader, val_data_loader, test_data_loader = train_val_test_data_loaders
        logger.info(
            f"train valid test splits:" +
            f" {len(train_data_loader.dataset.samples) if train_data_loader else None}," +
            f" {len(val_data_loader.dataset.samples) if val_data_loader else None}," +
            f" {len(test_data_loader.dataset.samples) if test_data_loader else None}")

        # MODEL
        if load_model_from_dir:
            # Load model from file
            model_path = os.path.join(load_model_from_dir, 'model.state')
            image_model = ThyroidClassificationModel(base_model).load_model(model_path).to(Config.available_device)
        else:
            image_model = ThyroidClassificationModel(base_model).to(Config.available_device)

        if train_model_flag:
            # TRAIN
            transformation = get_transformation(augmentation=augmentation, base_dataset=adaptation_sample_dataset)
            train_dataset = cast(ThyroidDataset, train_data_loader.dataset)
            train_dataset.transform = transformation

            cec = nn.CrossEntropyLoss(weight=torch.tensor(train_dataset.class_weights).to(Config.available_device))
            optimizer = optim.Adam(image_model.parameters(), lr=Config.learning_rate)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=Config.decay_rate)

            val_acc_history = []
            train_acc_history = []
            train_y_preds = []
            train_y_targets = []
            best_epoch_val_acc = 0

            for epoch in range(Config.n_epoch):
                # variables to calculate train acc
                class_set = sorted(train_data_loader.dataset.class_to_idx_dict.values())

                for images, labels in tqdm(train_data_loader, colour="#0000ff"):
                    if len(images) >= Config.batch_size // 2:
                        image_model.train()
                        images = images.to(Config.available_device)
                        labels = labels.to(Config.available_device)
                        optimizer.zero_grad()
                        pred = image_model(images)
                        # pred label: torch.max(pred, 1)[1], labels
                        if _is_inception:
                            pred, aux_pred = pred
                            loss, aux_loss = cec(pred, labels), cec(aux_pred, labels)
                            loss = loss + 0.4 * aux_loss
                        else:
                            loss = cec(pred, labels)
                        loss.backward()
                        optimizer.step()

                        # train preds and labels
                        values, preds = torch.max(pred, 1)
                        train_y_preds.extend(preds.cpu())
                        train_y_targets.extend(labels.cpu())

                # Epoch level
                # validation data
                image_model.eval()

                train_cf_matrix = confusion_matrix(train_y_targets, train_y_preds, normalize="true")

                class_accuracies = [train_cf_matrix[c][c] for c in class_set]
                train_acc = sum(class_accuracies)
                train_acc /= len(class_set)

                train_acc = (100 * sum(class_accuracies) / len(class_set)).item()
                train_acc_history.append(train_acc)
                logger.info(f'Train|E:{epoch}|Balanced Accuracy:{round(train_acc, 4)}%,\n{train_cf_matrix}')

                val_acc, val_cf_matrix, _, val_loss = validate(image_model,
                                                               val_data_loader,
                                                               cec)
                val_acc = float(val_acc)
                val_acc_history.append(val_acc)
                logger.info(f'Val|E:{epoch}|Balanced Accuracy:{round(val_acc, 4)}%,\n{val_cf_matrix}')

                save_model = False
                is_last_epoch = epoch == Config.n_epoch
                is_a_better_epoch = val_acc >= best_epoch_val_acc
                is_a_better_epoch &= abs(train_acc - val_acc) < Config.train_val_acc_max_distance_for_best_epoch
                if is_a_better_epoch or is_last_epoch:
                    save_model = True
                    calculate_test(image_model, epoch, test_data_loader, logger, config_name, show_tqdm=False)
                plot_and_save_model_per_epoch(epoch if save_model else None,
                                              image_model if save_model else None,
                                              val_acc_history,
                                              train_acc_history,
                                              [],
                                              [],
                                              config_label=config_name)
                my_lr_scheduler.step()
        else:
            # JUST EVALUATE
            calculate_test(image_model, 0, test_data_loader, logger, config_name,
                           show_tqdm=True)
    except Exception as e:
        print(e)
        logger.error(str(e))
        raise e


def load_datasets(datasets_folders, test_percent=Config.test_percent, val_percent=Config.val_percent, sample_percent=1,
                  is_nci_per_slide=False):
    if is_nci_per_slide:
        l_train, l_val, l_test = CustomFragmentLoader(
            datasets_folders).national_cancer_image_and_labels_splitter_per_slide(
            test_percent=test_percent,
            val_percent=val_percent)
    else:
        l_train, l_val, l_test = CustomFragmentLoader(datasets_folders).load_image_path_and_labels_and_split(
            test_percent=test_percent,
            val_percent=val_percent)

    l_train = random.choices(l_train, k=int(sample_percent * len(l_train)))
    l_val = random.choices(l_val, k=int(sample_percent * len(l_val)))
    l_test = random.choices(l_test, k=int(sample_percent * len(l_test)))

    l_train_ds = ThyroidDataset(l_train, Config.class_idx_dict)
    l_val_ds = ThyroidDataset(l_val, Config.class_idx_dict)
    l_test_ds = ThyroidDataset(l_test, Config.class_idx_dict)

    l_train_data_loader = None
    if l_train:
        l_train_data_loader = DataLoader(l_train_ds, batch_size=Config.batch_size, shuffle=True)
    l_val_data_loader = None
    if l_val:
        l_val_data_loader = DataLoader(l_val_ds, batch_size=Config.eval_batch_size, shuffle=True)
    l_test_data_loader = None
    if l_test:
        l_test_data_loader = DataLoader(l_test_ds, batch_size=Config.eval_batch_size, shuffle=True)

    return (l_train, l_val, l_test), (l_train_ds, l_val_ds, l_test_ds), (
        l_train_data_loader, l_val_data_loader, l_test_data_loader)


@torch.no_grad()
def evaluate_nci_dataset_per_slide(config_base_name, augmentation, base_model, data_loader,
                                   load_model_from_dir):
    config_name = f"{config_base_name}-{augmentation}-tumor-percent"

    logger = set_config_for_logger(config_name)
    logger.info(f"training config: {config_name}")
    _is_inception = type(base_model) == torchvision.models.inception.Inception3
    logger.info(
        f"test:" +
        f" {len(data_loader.dataset.samples) if data_loader else None}")

    # MODEL
    # Load model from file
    model_path = os.path.join(load_model_from_dir, 'model.state')
    model = ThyroidClassificationModel(base_model).load_model(model_path).to(Config.available_device)

    y_positive_scores = []
    slides_preds = {}
    slide_labels = {}
    for images, (labels, slides) in tqdm(data_loader):
        images = images.to(Config.available_device)

        x = model(images, validate=True).cpu()
        preds = x[:, 1]
        logger.info("zero and 1000 percent")
        logger.info(x[:, 0])
        logger.info(x[:, 1])
        for row_index in range(len(labels)):
            slide_id = slides[row_index]
            slide_label = labels[row_index]
            slide_labels[slide_id] = slide_label
            slides_preds[slide_id] = slides_preds.get(slide_id, []) + [preds[row_index].item()]
        y_positive_scores += x[:, 1].cpu()

    y_targets = []
    y_preds = []
    for key, value in slides_preds.items():
        slides_preds[key] = (sum(slides_preds[key]) / len(slides_preds[key])) * 100
        y_preds.append(slides_preds[key])
        y_targets.append(int(slide_labels[key]))

    y_targets_rounded = [int(round(x / 100, 1) * 100) for x in y_targets]
    y_preds_rounded = [int(round(x / 100, 1) * 100) for x in y_preds]
    cf_matrix = confusion_matrix(y_targets_rounded, y_preds_rounded, labels=Config.class_names, normalize="true")

    class_accuracies = [cf_matrix[c][c] for c in range(len(cf_matrix))]
    class_weights = [sum(cf_matrix[c]) for c in range(len(cf_matrix))]
    acc = sum([class_accuracies[i] * class_weights[i] for i in range(len(class_accuracies))])
    acc /= sum(class_weights)
    # TN|FN
    # FP|TP
    # fpr, tpr, _ = roc_curve(y_targets, y_positive_scores)
    # auc = roc_auc_score(y_targets, y_positive_scores)
    logger.info(f"target rounded:{y_targets_rounded}")
    logger.info(f"pred rounded:{y_preds_rounded}")
    logger.info(f"Results| acc:{acc * 100}\ncf:{cf_matrix}")
    return acc * 100, cf_matrix


##########
## Runs ##
##########
# train_phase block
if __name__ == '__main__' and Config.train_phase:
    _, (train_ds, _, _), (train_data_loader, val_data_loader, test_data_loader) = load_datasets(
        ["national_cancer_institute"],
        sample_percent=1)

    # Domain adaptation dataset on small real datasets
    # _, (_, _, domain_sample_test_dataset), _ = load_datasets(["stanford_tissue_microarray",
    #                                                           "papsociaty"],
    #                                                          sample_percent=0.5,
    #                                                          test_percent=100,
    #                                                          val_percent=0)

    for c_base_name, model, augmentations in [
        (f"resnet101_{Config.learning_rate}_{Config.decay_rate}_nci_final",
         torchvision.models.resnet101(pretrained=True, progress=True), [
             "mixup",
             # "jit",
             # "fda",
             # "jit-fda-mixup",
             # "shear",
             # "std"
         ]),
    ]:
        for aug in augmentations:
            Config.reset_random_seeds()
            train_model(model, c_base_name, (train_data_loader, val_data_loader, test_data_loader),
                        augmentation=aug, adaptation_sample_dataset=train_ds)
# evaluate_phase block
if __name__ == '__main__' and Config.evaluate_phase:
    # Main data
    Config.class_names = [i for i in range(101)]
    Config.class_idx_dict = {i: i for i in range(101)}
    _, (train_ds, _, _), (_, _, test_data_loader) = load_datasets(
        ["national_cancer_institute",
         ],
        sample_percent=1, test_percent=100, val_percent=0, is_nci_per_slide=True)

    for c_base_name, model, aug_best_epoch_list in [
        (f"resnet101_{Config.learning_rate}_{Config.decay_rate}_nci_eval",
         torchvision.models.resnet101(pretrained=True, progress=True), [
             ("mixup", "train_state/resnet101_0.0001_1_nci_final-mixup-BENIGN,MALIGNANT/epoch-19/"),
         ]),
        # (f"resnet101_{Config.learning_rate}_{Config.decay_rate}_test_nci_eval",
        #  torchvision.models.resnet101(pretrained=True, progress=True), [
        #      ("fda",
        #       "train_state/runs_0.0001_1_nic_test_benign_mal/resnet101_0.0001_1_nci-fda-BENIGN,MALIGNANT/epoch-3/"),
        #      ("mixup",
        #       "train_state/runs_0.0001_1_nic_test_benign_mal/resnet101_0.0001_1_nci-mixup-BENIGN,MALIGNANT/epoch-3/"),
        #      ("jit",
        #       "train_state/runs_0.0001_1_nic_test_benign_mal/resnet101_0.0001_1_nci-jit-BENIGN,MALIGNANT/epoch-3/"),
        #      ("jit-fda-mixup",
        #       "train_state/runs_0.0001_1_nic_test_benign_mal/resnet101_0.0001_1_nci-jit-fda-mixup-BENIGN,MALIGNANT/epoch-3/"),
        #  ]),

    ]:
        for aug, best_epoch in aug_best_epoch_list:
            Config.reset_random_seeds()
            evaluate_nci_dataset_per_slide(c_base_name, aug, model, test_data_loader,
                                           load_model_from_dir=best_epoch)

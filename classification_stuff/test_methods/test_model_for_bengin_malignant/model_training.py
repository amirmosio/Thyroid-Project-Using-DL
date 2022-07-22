import os
from typing import cast

import matplotlib.pyplot as plt
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
def validate(model, data_loader, loss_function=None):
    class_set = sorted(data_loader.dataset.class_to_idx_dict.values())

    loss_values = []
    y_preds = []
    y_targets = []

    for images, labels in data_loader:
        images = images.to(Config.available_device)
        labels = labels.to(Config.available_device)
        x = model(images, validate=True)
        if loss_function:
            loss_values.append(loss_function(x, labels))
        values, preds = torch.max(x, 1)

        y_preds += preds
        y_targets += labels

    cf_matrix = confusion_matrix(y_targets, y_preds, normalize="true")

    class_accuracies = [cf_matrix[c][c] for c in class_set]
    acc = sum(class_accuracies)
    acc /= len(class_set)
    # TN|FN
    # FP|TP
    fpr, tpr, _ = roc_curve(y_targets, y_preds)
    auc = roc_auc_score(y_targets, y_preds)
    if loss_function:
        loss = sum(loss_values)
        loss /= len(loss_values)
        return acc * 100, cf_matrix, (fpr, tpr, auc), loss
    return acc * 100, cf_matrix, (fpr, tpr, auc)


def plot_and_save_model_per_epoch(epoch,
                                  model_to_save,
                                  val_acc_list,
                                  train_acc_list,
                                  val_loss_list,
                                  train_loss_list,
                                  train_fpr,
                                  train_tpr,
                                  train_auc_score,
                                  val_fpr,
                                  val_tpr,
                                  val_auc_score,
                                  config_label):
    trains_state_dir = "./train_state"
    if not os.path.isdir(trains_state_dir):
        os.mkdir(trains_state_dir)
    config_train_dir = os.path.join(trains_state_dir, config_label)
    if not os.path.isdir(config_train_dir):
        os.mkdir(config_train_dir)

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

    save_state_dir = os.path.join(config_train_dir, f"epoch-{epoch}")
    if not os.path.isdir(save_state_dir):
        os.mkdir(save_state_dir)

    fig_save_path = os.path.join(config_train_dir, "val_train_auc.jpeg")
    plt.plot(val_fpr, val_tpr, label="validation, auc=" + str(val_auc_score))
    plt.plot(train_fpr, train_tpr, label="train, auc=" + str(train_auc_score))
    plt.legend(loc="lower right")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig(fig_save_path)
    plt.clf()

    model_save_path = os.path.join(save_state_dir, "model.state")
    model_to_save.save_model(model_save_path)


def train_model(base_model, config_base_name, train_val_test_data_loaders, augmentation):
    config_name = f"{config_base_name}-{augmentation}-{','.join(class_idx_dict.keys())}"

    logger = set_config_for_logger(config_name)
    logger.info(f"training config: {config_name}")
    try:
        _is_inception3 = type(base_model) == torchvision.models.inception.Inception3

        image_model = ThyroidClassificationModel(base_model).to(Config.available_device)
        transformation = get_transformation(augmentation=augmentation, base_data_loader=train_ds)

        train_data_loader, val_data_loader, test_data_loader = train_val_test_data_loaders
        cast(ThyroidDataset, train_data_loader.dataset).transform = transformation

        cec = nn.CrossEntropyLoss(weight=torch.tensor(train_ds.class_weights).to(Config.available_device))
        optimizer = optim.Adam(image_model.parameters(), lr=Config.learning_rate)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=Config.decay_rate)

        val_acc_history = [0]
        train_acc_history = [0]

        for epoch in range(Config.n_epoch):
            # variables to calculate train acc
            class_set = sorted(train_data_loader.dataset.class_to_idx_dict.values())
            train_y_preds = []
            train_y_targets = []

            for images, labels in tqdm(train_data_loader, colour="#0000ff"):
                image_model.train()
                images = images.to(Config.available_device)
                labels = labels.to(Config.available_device)
                optimizer.zero_grad()
                pred = image_model(images)
                # pred label: torch.max(pred, 1)[1], labels
                if _is_inception3:
                    pred, aux_pred = pred
                    loss, aux_loss = cec(pred, labels), cec(aux_pred, labels)
                    loss = loss + 0.4 * aux_loss
                else:
                    loss = cec(pred, labels)
                loss.backward()
                optimizer.step()

                # train preds and labels
                values, preds = torch.max(pred, 1)
                train_y_preds.extend(preds)
                train_y_targets.extend(labels)

            # Epoch level
            # validation data
            image_model.eval()

            train_cf_matrix = confusion_matrix(train_y_targets, train_y_preds, normalize="true")

            class_accuracies = [train_cf_matrix[c][c] for c in class_set]
            train_acc = sum(class_accuracies)
            train_acc /= len(class_set)
            train_FPR, train_TPR, _ = roc_curve(train_y_targets, train_y_preds)
            train_auc_score = roc_auc_score(train_y_targets, train_y_preds)

            train_acc = (100 * sum(class_accuracies) / len(class_set)).item()
            train_acc_history.append(train_acc)
            logger.info(f'Train|E:{epoch + 1}|Balanced Accuracy:{round(train_acc, 4)}%,\n{train_cf_matrix}')

            val_acc, val_cf_matrix, (val_FPR, val_TPR, val_auc_score), val_loss = validate(image_model,
                                                                                           val_data_loader,
                                                                                           cec)
            val_acc = float(val_acc)
            val_acc_history.append(val_acc)
            logger.info(f'Val|E:{epoch + 1}|Balanced Accuracy:{round(val_acc, 4)}%,\n{val_cf_matrix}')

            plot_and_save_model_per_epoch(epoch,
                                          image_model,
                                          val_acc_history,
                                          train_acc_history,
                                          [],
                                          [],
                                          train_fpr=train_FPR,
                                          train_tpr=train_TPR,
                                          train_auc_score=train_auc_score,
                                          val_fpr=val_FPR,
                                          val_tpr=val_TPR,
                                          val_auc_score=val_auc_score,
                                          config_label=config_name)
            my_lr_scheduler.step()
    except Exception as e:
        print(e)
        logger.error(str(e))
        raise e
    else:
        # Test acc
        image_model.eval()
        test_acc, test_c_acc, (test_FPR, test_TPR, test_auc_score) = validate(image_model, test_data_loader)
        test_acc = float(test_acc)
        logger.info(f'Test|Accuracy:{round(test_acc, 4)}, {test_c_acc}%')


if __name__ == '__main__':
    class_idx_dict = {"BENIGN": 0, "MALIGNANT": 1}
    datasets_folder = ["stanford_tissue_microarray", "papsociaty"]
    train, val, test = CustomFragmentLoader(datasets_folder).load_image_path_and_labels_and_split(
        test_percent=Config.test_percent,
        val_percent=Config.val_percent)
    test_ds = ThyroidDataset(test, class_idx_dict)
    val_ds = ThyroidDataset(val, class_idx_dict)
    train_ds = ThyroidDataset(train, class_idx_dict)

    train_data_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
    val_data_loader = DataLoader(val_ds, batch_size=Config.eval_batch_size, shuffle=True)
    test_data_loader = DataLoader(test_ds, batch_size=Config.eval_batch_size, shuffle=True)

    for config_base_name, model in [
        ("resnet18_lr_decay", torchvision.models.resnet18(pretrained=True, progress=True)),
        ("resnet34_lr_decay", torchvision.models.resnet34(pretrained=True, progress=True)),
        ("inception_v3_lr_decay", torchvision.models.inception_v3(pretrained=True, progress=True)),
    ]:
        for aug in [
            "jit",
            "jit-nrs",
            "fda",
            "mixup"
        ]:
            train_model(model, config_base_name, (train_data_loader, val_data_loader, test_data_loader),
                        augmentation=aug)

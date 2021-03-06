import os
from typing import cast

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
def validate(model, data_loader, loss_function=None):
    class_set = sorted(data_loader.dataset.class_to_idx_dict.values())

    loss_values = []
    y_preds = []
    y_targets = []
    y_positive_scores = []

    for images, labels in data_loader:
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

    model_save_path = os.path.join(save_state_dir, "model.state")
    model_to_save.save_model(model_save_path)


def save_auc_roc_chart_for_test(test_fpr, test_tpr, test_auc_score, config_label, epoch):
    trains_state_dir, config_train_dir, save_dir = get_save_state_dirs(config_label, epoch)
    fig_save_path = os.path.join(save_dir, "test_roc.jpeg")
    plt.plot(test_fpr, test_tpr, label="test, auc=" + str(test_auc_score))
    plt.legend(loc="lower right")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig(fig_save_path)
    plt.clf()


def train_model(base_model, config_base_name, train_val_test_data_loaders, augmentation,
                load_model_from_epoch_and_run_test=None):
    config_name = f"{config_base_name}-{augmentation}-{','.join(Config.class_idx_dict.keys())}"

    logger = set_config_for_logger(config_name)
    logger.info(f"training config: {config_name}")
    try:
        _is_inception = type(base_model) == torchvision.models.inception.Inception3

        transformation = get_transformation(augmentation=augmentation, base_data_loader=train_ds)

        train_data_loader, val_data_loader, test_data_loader = train_val_test_data_loaders
        cast(ThyroidDataset, train_data_loader.dataset).transform = transformation

        if load_model_from_epoch_and_run_test is None:
            image_model = ThyroidClassificationModel(base_model).to(Config.available_device)

            cec = nn.CrossEntropyLoss(weight=torch.tensor(train_ds.class_weights).to(Config.available_device))
            optimizer = optim.Adam(image_model.parameters(), lr=Config.learning_rate)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=Config.decay_rate)

            val_acc_history = [0]
            train_acc_history = [0]
            train_y_preds = []
            train_y_targets = []
            for epoch in range(Config.n_epoch):
                # variables to calculate train acc
                class_set = sorted(train_data_loader.dataset.class_to_idx_dict.values())

                for images, labels in tqdm(train_data_loader, colour="#0000ff"):
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

                plot_and_save_model_per_epoch(epoch,
                                              image_model,
                                              val_acc_history,
                                              train_acc_history,
                                              [],
                                              [],
                                              config_label=config_name)
                my_lr_scheduler.step()
        else:
            # Load model from file
            save_dir = get_save_state_dirs(config_name, load_model_from_epoch_and_run_test)[2]
            model_path = os.path.join(save_dir, 'model.state')
            image_model = ThyroidClassificationModel(base_model).load_model(model_path).to(Config.available_device)
    except Exception as e:
        print(e)
        logger.error(str(e))
        raise e
    else:
        # Test acc
        image_model.eval()
        test_acc, test_c_acc, (test_FPR, test_TPR, test_auc_score) = validate(image_model, test_data_loader)
        test_acc = float(test_acc)

    if load_model_from_epoch_and_run_test is not None:
        save_auc_roc_chart_for_test(test_FPR, test_TPR, test_auc_score, config_name, load_model_from_epoch_and_run_test)
        logger.info(f'Test|Epoch:{load_model_from_epoch_and_run_test}|Accuracy:{round(test_acc, 4)}, {test_c_acc}%')
    else:
        save_auc_roc_chart_for_test(test_FPR, test_TPR, test_auc_score, config_name, epoch)
        logger.info(f'Test|Epoch:{epoch}|Accuracy:{round(test_acc, 4)}, {test_c_acc}%')


if __name__ == '__main__':
    datasets_folder = ["stanford_tissue_microarray", "papsociaty"]
    train, val, test = CustomFragmentLoader(datasets_folder).load_image_path_and_labels_and_split(
        test_percent=Config.test_percent,
        val_percent=Config.val_percent)
    test_ds = ThyroidDataset(test, Config.class_idx_dict)
    val_ds = ThyroidDataset(val, Config.class_idx_dict)
    train_ds = ThyroidDataset(train, Config.class_idx_dict)

    train_data_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
    val_data_loader = DataLoader(val_ds, batch_size=Config.eval_batch_size, shuffle=True)
    test_data_loader = DataLoader(test_ds, batch_size=Config.eval_batch_size, shuffle=True)

    for config_base_name, model, augmentations in [
        (f"resnet18_{Config.learning_rate}*{Config.decay_rate}", torchvision.models.resnet18(pretrained=True, progress=True), [
            "jit",
            "jit-nrs",
            "fda",
            "mixup",
            "jit-fda-mixup"
        ]),
        (f"resnet101_{Config.learning_rate}*{Config.decay_rate}",
         torchvision.models.resnet101(pretrained=True, progress=True), [
             "jit",
             "jit-nrs",
             "fda",
             "mixup",
             "jit-fda-mixup"
         ]),
        ("resnet34_lr_decay", torchvision.models.resnet34(pretrained=True, progress=True), [
            "jit",
            "jit-nrs",
            "fda",
            "mixup",
            "jit-fda-mixup"
        ]),
        ("inception_v3_lr_decay", torchvision.models.inception_v3(pretrained=True, progress=True), [
            "jit",
            "jit-nrs",
            "fda",
            "mixup",
            "jit-fda-mixup"
        ]),
        ("inception_v4_lr_decay", timm.create_model('inception_v4', pretrained=True), [
            "jit",
            "jit-nrs",
            "fda",
            "mixup",
            "jit-fda-mixup"
        ])
    ]:
        for aug in augmentations:
            Config.reset_random_seeds()
            train_model(model, config_base_name, (train_data_loader, val_data_loader, test_data_loader),
                        augmentation=aug)

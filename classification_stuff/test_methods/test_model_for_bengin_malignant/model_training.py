import os

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification_stuff.config import Config
from fragment_splitter import CustomFragmentLoader
from thyroid_dataset import ThyroidDataset
from thyroid_ml_model import ThyroidClassificationModel
from transformation import get_transformation


@torch.no_grad()
def validate(model, data_loader):
    class_set = sorted(data_loader.dataset.class_to_idx_dict.values())
    class_total_count = {}
    class_correct_count = {}

    e = 0.00001
    for images, labels in data_loader:
        images = images.to(Config.available_device)
        labels = labels.to(Config.available_device)
        x = model(images, validate=True)
        values, preds = torch.max(x, 1)
        for c in class_set:
            class_correct_count[c] = class_correct_count.get(c, 0) + ((preds == labels) * (labels == c)).sum()
            class_total_count[c] = class_total_count.get(c, 0) + (labels == c).sum()
    class_accuracies = [class_correct_count[c] / (class_total_count[c] + e) for c in class_set]
    acc = sum(class_accuracies)
    acc /= len(class_set)
    return acc * 100, class_accuracies


def train_model(base_model, model_name, sort_batch=False, augmentation="min"):
    _is_inception3 = type(base_model) == torchvision.models.inception.Inception3

    class_idx_dict = {"BENIGN": 0, "MALIGNANT": 1}
    datasets_folder = ["stanford_tissue_microarray", "papsociaty"]

    config_name = f"{model_name}-{augmentation}-{','.join(class_idx_dict.keys())}-{','.join(datasets_folder)}"

    logger = set_config_for_logger(config_name)
    logger.info(f"training config: {config_name}")
    try:
        image_model = ThyroidClassificationModel(base_model).to(Config.available_device)
        transformation = get_transformation(augmentation="min")

        train, val, test = CustomFragmentLoader(datasets_folder).load_image_path_and_labels_and_split(
            test_percent=Config.test_percent,
            val_percent=Config.val_percent)
        train_ds = ThyroidDataset(train, class_idx_dict, transform=transformation)
        test_ds = ThyroidDataset(test, class_idx_dict)
        val_ds = ThyroidDataset(val, class_idx_dict)

        train_data_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
        val_data_loader = DataLoader(val_ds, batch_size=Config.eval_batch_size, shuffle=True)
        test_data_loader = DataLoader(test_ds, batch_size=Config.eval_batch_size, shuffle=True)

        cec = nn.CrossEntropyLoss(weight=torch.tensor(train_ds.class_weights).to(Config.available_device))
        optimizer = optim.Adam(image_model.parameters(), lr=Config.learning_rate)
        val_acc_history = []
        test_acc_history = []

        i = -1
        val_acc = 0
        test_acc = 0
        for e in range(Config.n_epoch):
            for images, labels in tqdm(train_data_loader, colour="#0000ff"):
                # if type(base_model) == torchvision.models.inception.Inception3:
                # if image_model
                image_model.train()
                i += 1
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
                if (i + 1) % Config.n_print == 0:
                    image_model.eval()
                    val_acc, val_c_acc = validate(image_model, val_data_loader)
                    val_acc = float(val_acc)
                    logger.info(f'Val|E:{e + 1}, B:{i + 1} Loss:{float(loss.data)} Accuracy:{val_acc}%, {val_c_acc}')

                    val_acc_history.append(val_acc)
                    test_acc_history.append(test_acc)

            image_model.eval()
            test_acc, test_c_acc = validate(image_model, test_data_loader)
            test_acc = float(test_acc)
            logger.info(f'Test|Epoch:{e + 1}, Accuracy: {test_acc}, {test_c_acc}%')

            plot_and_save_model_per_epoch(e, image_model, val_acc_history, test_acc_history, config_name)
    except Exception as e:
        print(e)
        logger.info(str(e))


def set_config_for_logger(config_label):
    import logging
    trains_state_dir = "./train_state"
    if not os.path.isdir(trains_state_dir):
        os.mkdir(trains_state_dir)
    config_train_dir = os.path.join(trains_state_dir, config_label)
    if not os.path.isdir(config_train_dir):
        os.mkdir(config_train_dir)
    log_file = os.path.join(config_train_dir, "console.log")
    logger = logging.getLogger(config_label)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s|%(levelname)s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


def plot_and_save_model_per_epoch(epoch, model, val_acc_list, test_acc_list, config_label):
    trains_state_dir = "./train_state"
    if not os.path.isdir(trains_state_dir):
        os.mkdir(trains_state_dir)
    config_train_dir = os.path.join(trains_state_dir, config_label)
    if not os.path.isdir(config_train_dir):
        os.mkdir(config_train_dir)

    fig_save_path = os.path.join(config_train_dir, "val_test_acc.jpeg")
    plt.plot(range(len(val_acc_list)), val_acc_list, label="val")
    plt.plot(range(len(test_acc_list)), test_acc_list, label="test")
    plt.legend(loc="lower right")
    plt.savefig(fig_save_path)
    plt.clf()

    save_state_dir = os.path.join(config_train_dir, f"epoch-{epoch}")
    if not os.path.isdir(save_state_dir):
        os.mkdir(save_state_dir)
    model_save_path = os.path.join(save_state_dir, "model.state")
    model.save_model(model_save_path)


if __name__ == '__main__':
    for model_name, model in [
        ("resnet18", torchvision.models.resnet18(pretrained=True, progress=True)),
        ("resnet34", torchvision.models.resnet34(pretrained=True, progress=True)),
        ("inception_v3", torchvision.models.inception_v3(pretrained=True, progress=True)),
        ("vgg19", torchvision.models.vgg19(pretrained=True, progress=True)),
    ]:
        for aug in ["fda",
                    "std",
                    "mixup"
                    ]:
            train_model(model, model_name, augmentation=aug)

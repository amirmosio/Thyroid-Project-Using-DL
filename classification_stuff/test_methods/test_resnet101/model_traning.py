import torch
import torchvision
from torch import nn, optim

from classification_stuff.config import n_epoch, learning_rate, available_device, n_print
from classification_stuff.data_loader import DataLoaderUtil


class ThyroidClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_net_101 = torchvision.models.resnet101(pretrained=True, progress=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        res_net_output = self.res_net_101(x)
        return self.classifier(res_net_output)


# def validate(model, data):
#     # To get validation accuracy = (correct/total)*100.
#     total = 0
#     correct = 0
#     for i, (images, labels) in enumerate(data):
#         images = var(images.cuda())
#         x = model(images)
#         value, pred = torch.max(x, 1)
#         pred = pred.data.cpu()
#         total += x.size(0)
#         correct += torch.sum(pred == labels)
#     return correct * 100. / total
#
#
# def train_model(image_model, sort_batch=False):
#     bio_atlas_data_dir = "../database_crawlers/bio_atlas_at_jake_gittlen_laboratories/data/"
#     data_loader = DataLoaderUtil.final_data_loader_in_batches_test_validate_train(bio_atlas_data_dir)
#     cec = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(image_model.parameters(), lr=learning_rate)
#     acc_history = []
#     for e in range(n_epoch):
#         for i, (thyroid_types, frags) in enumerate(data_loader):
#             frags = torch.var(frags.to(available_device))
#             labels = torch.var(labels.to(available_device))
#             optimizer.zero_grad()
#             pred = image_model(frags)
#             loss = cec(pred, labels)
#             loss.backward()
#             optimizer.step()
#             if (i + 1) % n_print == 0:
#                 accuracy = float(validate(image_model, val_dl))
#                 print('Epoch :', e + 1, 'Batch :', i + 1, 'Loss :', float(loss.data), 'Accuracy :', accuracy, '%')
#                 acc_history.append(accuracy)
#         return acc_history

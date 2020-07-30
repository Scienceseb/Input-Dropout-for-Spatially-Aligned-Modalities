import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import torch
import matplotlib.pyplot as plt


class History:

    def __init__(self):
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }

    def save(self, train_acc, val_acc, train_loss, val_loss):
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)

    def display(self):
        epoch = len(self.history['train_acc'])
        epochs = [x for x in range(1, epoch + 1)]

        fig, axes = plt.subplots(2, 1)
        plt.tight_layout()

        axes[0].set_title('Train accuracy')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].plot(epochs, self.history['train_acc'], label='Train')
        axes[0].plot(epochs, self.history['val_acc'], label='Validation')
        axes[0].legend()

        axes[1].set_title('Train loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].plot(epochs, self.history['train_loss'], label='Train')
        axes[1].plot(epochs, self.history['val_loss'], label='Validation')
        plt.show()

softmax=False

def validate(model, val_loader, use_gpu=False):
    true = []
    pred = []
    val_loss = []

    criterion = nn.CrossEntropyLoss()
    model.eval()

    with torch.no_grad():
        for j, batch in enumerate(val_loader):

            inputs, targets = batch

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            output = model(inputs)


            predictions = output.max(dim=1)[1]

            val_loss.append(criterion(output, targets).item())

            true.extend(targets.data.cpu().numpy().tolist())
            pred.extend(predictions.data.cpu().numpy().tolist())

    return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)


def train(model, train_loader,val_loader, n_epoch, learning_rate, use_gpu=False):
    history = History()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(n_epoch):
        model.train()
        for j, batch in enumerate(train_loader):

            inputs, targets = batch

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            output = model(inputs)

            loss=criterion(output, targets)

            loss.backward()
            optimizer.step()

        train_acc, train_loss = validate(model, train_loader, use_gpu)
        val_acc, val_loss = validate(model, val_loader, use_gpu)
        history.save(train_acc, val_acc, train_loss, val_loss)
        # print('Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f}'.format(i,
        #                                                                                                       train_acc,
        #                                                                                                       val_acc,
        #                                                                                                       train_loss,
        #                                                                                                       val_loss))
    return history


def test(model, loader, use_gpu=False):
    return validate(model, loader, use_gpu)
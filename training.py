import torch
import torch.nn as nn
import torch.optim as optim

from Classifier import Classifier
from cifar import train_loader


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    classifier = Classifier()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        classifier.parameters(),
        lr=0.001,
        momentum=0.9
    )

    for epoch in range(2):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

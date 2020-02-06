import torch
import torch.nn as nn
import torch.optim as optim

from Classifier import Classifier
from helpers import get_train_set_and_loader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    _, train_loader = get_train_set_and_loader()

    classifier = Classifier()
    classifier.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        classifier.parameters(),
        lr=0.001,
        momentum=0.9
    )

    for epoch in range(2):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

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

    PATH = './model_storage/cifar_net.pth'
    torch.save(classifier.state_dict(), PATH)

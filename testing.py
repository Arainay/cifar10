import torch
import torchvision

from Classifier import Classifier
from classes import classes
from helpers import show_image, compute_accuracy, get_test_set_and_loader

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    _, test_loader = get_test_set_and_loader()

    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    show_image(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    classifier = Classifier()
    classifier.load_state_dict(torch.load('./model_storage/cifar_net.pth'))

    outputs = classifier(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    compute_accuracy(test_loader, classifier)

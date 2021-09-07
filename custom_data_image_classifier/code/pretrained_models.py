import torch
from torchvision import models, transforms
from PIL import Image


class WhatIsThat:
    """
    Class WhatIsThat:  Loads a ResNet101 or AlexNet that has been pretrained on
    ImageNet, and runs it with no training or fine tuning on an input user image.

    Sample usage:

        model = WhatIsThat()
        model.load_image('my_custom_image.jpg')
        model.run_resnet()
        model.identify_top5()

    `identify_top5()` returns the top 5 predicted classes and the probability
    of each class

    The model assumes that it is in a "code" folder, and there is a neighboring
    data folder with a file of classnames that can be changed/initializes when
    calling the class
    """

    def __init__(self, classnames_file="../data/imagenet_classes.txt"):
        self._load_models(classnames_file)
        self._image_transformations()
        self.resnet.eval()  # put resnet into eval mode

    def _load_models(self, classnames_file):
        self.alexnet = models.AlexNet()

        self.resnet = models.resnet101(pretrained=True)
        with open(classnames_file) as f:
            self.imagenet_classes = [line.strip() for line in f.readlines()]

    def _image_transformations(self):
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_image(self, filename):
        img = Image.open(filename)
        img.show()
        self._image_transformations()
        img_t = self.preprocess(img)
        self.img = torch.unsqueeze(img_t, 0)

    def run_resnet(self):  # image should be processed using load_image()
        self.resnet_output = self.resnet(self.img)
        self.percentage = (
            torch.nn.functional.softmax(self.resnet_output, dim=1)[0] * 100
        )

    def identify_best(self):
        _, index = torch.max(self.resnet_output, 1)
        return self.imagenet_classes[index[0]], self.percentage[index[0]].item()

    def identify_top5(self):
        _, index = torch.sort(self.resnet_output, descending=True)
        return [
            (self.imagenet_classes[idx], self.percentage[idx].item())
            for idx in index[0][0:5]
        ]


def main():
    classifier = WhatIsThat()
    filename = "../data/cat9.jpg"
    print(filename)
    classifier.load_image(filename)
    classifier.run_resnet()
    output = classifier.identify_top5()
    for i in output:
        print(i)


if __name__ == "__main__":
    main()

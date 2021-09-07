import torch
from torchvision import transforms
from PIL import Image

from resnet_generator import ResNetGenerator


def zebras(filename_output, filename="horse2zebra_0.4.0.pth"):
    netG = ResNetGenerator()

    model_weights = torch.load(filename)
    netG.load_state_dict(model_weights)
    preprocess = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

    # img = Image.open('horse.jpg')
    img = Image.open("horses3.jpg")

    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    batch_out = netG(batch_t)
    out_t = (batch_out.data.squeeze() + 1.0) / 2.0
    out_img = transforms.ToPILImage()(out_t)
    out_img.save(filename_output)


def main():
    zebras("zebra2.jpg", filename="horse2zebra_0.4.0.pth")


if __name__ == "__main__":
    main()

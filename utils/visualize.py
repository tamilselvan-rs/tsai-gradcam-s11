import matplotlib.pyplot as plt
import torchvision as vision

def visualize_transforms(data_loader):
    for images, _ in data_loader:
        print('images.shape:', images.shape)
        plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.imshow(vision.utils.make_grid(images, nrow=16).permute((1, 2, 0)))
        break
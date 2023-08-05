import matplotlib.pyplot as plt

def plot_results(train_acc, train_losses, test_acc, test_losses):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def plot_data(data_loader):
    batch_iter = iter(data_loader);
    batch_data, batch_label = next(batch_iter) 
    print("Batch Size {}".format(batch_data.shape));

    fig = plt.figure()

    for i in range(128):
        plt.subplot(13,10,i+1)
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        # plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])
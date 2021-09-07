from matplotlib import pyplot as plt


def plot_loss_accuracy(histories_):
    loss = histories_.history['loss']
    accuracy = histories_.history['accuracy']
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axs[0].plot(range(len(loss)), loss)
    axs[0].set(xlabel="Epochs", ylabel="Loss", title="Training loss")
    axs[1].plot(range(len(accuracy)), accuracy)
    axs[1].set(xlabel="Epochs", ylabel="Accuracy", title="Training accuracy")
    plt.show()
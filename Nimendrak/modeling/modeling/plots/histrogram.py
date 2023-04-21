import matplotlib.pyplot as plt

def histrogram(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = list(range(0, len(history.history['accuracy']) + 1))

    fig, ax1 = plt.subplots(figsize=(6, 6))

    # Plot accuracy on the first axis
    ax1.plot(epochs_range[1:], acc, label='Training Accuracy', color='royalblue')
    ax1.plot(epochs_range[1:], val_acc, label='Validation Accuracy', color='dodgerblue', linestyle='--')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([min(plt.ylim()), 1])
    ax1.set_title(f'Training and Validation Accuracy and Loss\nANNOE_v1.0')
    ax1.legend(
        loc="upper left",
        # bbox_to_anchor=(0.25, 0.5),
        # borderaxespad=0,
    )

    # Create a second y-axis to plot loss
    ax2 = ax1.twinx()
    ax2.plot(epochs_range[1:], loss, label='Training Loss', color='red')
    ax2.plot(epochs_range[1:], val_loss, label='Validation Loss', color='darkorange', linestyle='--')
    ax2.set_ylabel('Cross Entropy')
    ax2.set_ylim([0, 1.0])
    ax2.legend(
        loc="lower left",
        # bbox_to_anchor=(0.99, 0.5),
        # borderaxespad=0,
    )

    # Set the shared x-axis label
    ax1.set_xlabel('Epochs', loc='right')
    ax1.set_xlim([1, max(epochs_range)])

    ax1.autoscale(True)
    ax1.margins(0.05)

    # Set grid lines
    # ax1.grid(True)
    ax1.grid(which='major', color='#DDDDDD', linewidth=0.8)
    ax1.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    ax1.minorticks_on()

    plt.savefig('plots/histrogram.png', dpi=300, bbox_inches='tight')
    plt.show()

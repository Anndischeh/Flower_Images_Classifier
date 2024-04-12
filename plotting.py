import matplotlib.pyplot as plt
import random
"""
This class help us in visualizing result and evaluation 
"""

class ShowResult:
    def __init__(self):
        pass
    
    def visualize_training_results(self, history):
        train_acc = history.history['accuracy']
        valid_acc = history.history['val_accuracy']
        train_loss = history.history['loss']
        valid_loss = history.history['val_loss']
        train_auc = history.history['auc']
        valid_auc = history.history['val_auc']
        epochs = [i + 1 for i in range(len(train_acc))]

        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_acc, label='Training accuracy')
        plt.plot(epochs, valid_acc, label='Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Epochs vs Accuracy')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_auc, label='Training auc')
        plt.plot(epochs, valid_auc, label='Validation auc')
        plt.xlabel('Epochs')
        plt.ylabel('AUC score')
        plt.title('Epochs vs AUC')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, train_loss, label='Training loss')
        plt.plot(epochs, valid_loss, label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Epochs vs Loss')
        plt.legend()

        plt.show()
    
    def display_predictions(self, test_df, predictions):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))

        for i, ax in enumerate(axes.flat):
            idx = random.randint(0, test_df.shape[0])
            ax.imshow(plt.imread(test_df.FilePath.iloc[idx]))
            ax.set_title(f"True: {test_df.Labels.iloc[idx]}  \n Predicted: {test_df.Labels.iloc[idx]}")

        plt.tight_layout()
        plt.show()

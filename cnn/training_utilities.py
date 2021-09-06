''' Functions with create custom Callbacks, plot the learning curve at the end of training
and save the fitted model as either hdf5 or tf.SavedModel format '''

# Libraries
from tensorflow.keras import callbacks
from matplotlib import pyplot as plt
import datetime


def training_callbacks(ProgbarLogger, TensorBoard, CSVLogger, ModelCheckpoint, LearningRateScheduler):
    if ProgbarLogger:
        return callbacks.ProgbarLogger(count_mode="steps", stateful_metrics=None),
    if TensorBoard:
        return callbacks.TensorBoard(log_dir="./logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                     histogram_freq=1),  # Run with: tensorboard --logdir logs [data-folder]
    if CSVLogger:
        return callbacks.CSVLogger(f'training_log{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.csv',
                                   separator=",", append=False),
    if ModelCheckpoint:
        return callbacks.ModelCheckpoint(filepath='model_binary_Checkpoint.hdf5', save_best_only=True,
                                         save_weights_only=False, monitor='accuracy', mode='auto')
    if LearningRateScheduler:
        learning_rate = 0.01 * 1 / (1 + 0.0 * 1)
        return callbacks.LearningRateScheduler(learning_rate, verbose=1)


def scheduler(epoch, lr):
    # Set training parameters
    initial_learning_rate = 0.01
    epochs = 600

    decay = initial_learning_rate / epochs
    if epoch < 20:
        return lr
    else:
        return lr * 1 / (1 + decay * epoch)


def model_performance_plotting(record):
    # accuracy
    plt.style.use("ggplot")
    plt.plot(record.history['accuracy'])
    plt.plot(record.history['val_accuracy'])
    plt.title('Algorithm accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.savefig('model_accuracy.png')

    # loss
    plt.style.use("ggplot")
    plt.plot(record.history['loss'])
    plt.plot(record.history['val_loss'])
    plt.title('Algorithm loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.savefig('model_loss.png')


def save_classifier(fitted_model, format: str):
    if format == "tf_SavedModel":
        fitted_model.save(
            f'classifier_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}', overwrite=True)
        print("Trained model saved to disk in TensorFlow SavedModel format!")

    if format == "hdf5":
        fitted_model.save(
            f'classifier_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.hdf5', overwrite=True)
        print("Trained model saved to disk as hdf5 file!")

    return fitted_model

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from MRInet import CNN_SUBJECT_LEVEL

SAVE_DIR = './CNN_SUBJECT_LEVEL_RESULTS2/'

# HYPERPARAMETER SELECTIONS
batch_size = 10
tra_val_split = 0.8

epochs = [10, 20, 30]
learning_rates = [1e-8, 1e-6, 1e-4]

# FEATURE SELECTION
train_loss = np.zeros((len(epochs),
                       len(learning_rates)))

train_accu = np.zeros((len(epochs),
                       len(learning_rates)))

valid_loss = np.zeros((len(epochs),
                       len(learning_rates)))

valid_accu = np.zeros((len(epochs),
                       len(learning_rates)))

test_loss = np.zeros((len(epochs),
                      len(learning_rates)))

test_accu = np.zeros((len(epochs),
                      len(learning_rates)))


cnn = CNN_SUBJECT_LEVEL()
cnn.get_data(balanced=1, tra_val_split=tra_val_split, use_validation=True)
cnn.data_augmentation({'rotation':5})
cnn.set_tf_datasets(batch_size=batch_size)
cnn.build_model()

for m in range(len(epochs)):
    epoch = epochs[m]
    for n in range(len(learning_rates)):
        lr = learning_rates[n]

        tl, ta, vl, va = cnn.run(lr=lr, epochs=epoch)

        train_loss[m, n] = tl
        train_accu[m, n] = ta
        valid_loss[m, n] = vl
        valid_accu[m, n] = va

        # Test predictions to save time, 
        # but make choice based on validation
        t_loss, t_accu = cnn.test()
        test_loss[m, n] = t_loss
        test_accu[m, n] = t_accu

best_inds_collect = []

# Choose result from above with best validation accuracy
best_inds = np.unravel_index(np.argmax(valid_accu, axis=None), valid_accu.shape)
best_inds_collect.append(best_inds)
print(best_inds)

# Get test results
m, n = best_inds
epoch = epochs[m]
lr = learning_rates[n]

#cnn = ROI_SUBJECT_LEVEL()
#cnn.get_data(balanced=1, tra_val_split=tra_val_split, use_validation=False)
#cnn.data_augmentation({'rotation':5})
#cnn.set_tf_datasets(batch_size=batch_size)

#cnn.build_model()
#tl, ta, vl, va = cnn.run(lr=lr, epochs=epoch)

train_loss_final = train_loss[m, n]
train_accu_final = train_accu[m, n]
valid_loss_final = valid_loss[m, n]
valid_accu_final = valid_accu[m, n]
test_loss_final = test_loss[m, n]
test_accu_final = test_accu[m, n]

#t_loss, t_accuracy = cnn.test()
#test_loss[i] = t_loss
#test_accu[i] = t_accuracy

#f1 = plt.figure()
#ax1 = f1.add_subplot(111)
#ax1.plot(np.arange(1, num_masks+1), train_accu_final, '-o')
#ax1.plot(np.arange(1, num_masks+1), valid_accu_final, '-o')
#ax1.plot(np.arange(1, num_masks+1), test_accu, '-o')
#ax1.set_xlabel('Number of ROIs')
#ax1.set_ylabel('Accuracy')
#ax1.set_ylim([0.5, 1])
#ax1.legend(('training', 'validation', 'test'))
#
#plt.savefig(SAVE_DIR+'ROI_CNN_results_accuracy.pdf')
#
#f2 = plt.figure()
#ax2 = f2.add_subplot(111)
#ax2.plot(np.arange(1, num_masks+1), train_loss_final, '-o')
#ax2.plot(np.arange(1, num_masks+1), valid_loss_final, '-o')
#ax2.plot(np.arange(1, num_masks+1), test_loss, '-o')
#ax2.set_xlabel('Number of ROIs')
#ax2.set_ylabel('Loss')
#ax2.legend(('training', 'validation', 'test'))
    
#plt.savefig(SAVE_DIR+'ROI_CNN_results_loss.pdf')

print((train_loss_final, train_accu_final))
print((valid_loss_final, valid_accu_final))
print((test_loss_final, test_accu_final))

save_dict = {'train_loss_final': train_loss_final,
             'valid_loss_final': valid_loss_final,
             'train_accu_final': train_accu_final,
             'valid_accu_final': valid_accu_final,
             'test_loss_final': train_accu_final,
             'test_accu_final': test_accu_final,
             'test_loss': test_loss,
             'test_accu': test_accu,
             'train_loss': train_loss,
             'valid_loss': valid_loss,
             'train_accu': train_accu,
             'valid_accu': valid_accu,
             'best_inds_collect': best_inds_collect}

scipy.io.savemat(SAVE_DIR+'CNN_SUBJECT_LEVEL_results.mat', save_dict)

#plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from MRInet import ROI_CNN

SAVE_DIR = './ROI_CNN_RESULTS2/'

# MOST PREDICTIVE ATLAS MASKS -- FROM SVM RESULTS
mask_selectors = [2,3,28,9,43,10,25,33,13, 
                 12,40,31,32,41,42,21,16,39,
                 44,1,14,11,45,4,22,27,29, 
                 17,30,5,24,6,37,36,38,7, 
                 3,18,35,15,20,34,19,8,26]
mask_selectors.reverse()
# Reverse orders from most to least predictive

# HYPERPARAMETER SELECTIONS
num_masks = 6
activations = ['relu']
small_filters = [1, 2]
big_filters = [2, 4]
batch_size = 10
tra_val_split = 0.8

epochs = [30]
learning_rates = [1e-5, 1e-3]

# FEATURE SELECTION
train_loss = np.zeros((len(activations),
                       len(small_filters),
                       len(big_filters),
                       len(epochs),
                       len(learning_rates),
                       num_masks))

train_accu = np.zeros((len(activations),
                       len(small_filters),
                       len(big_filters),
                       len(epochs),
                       len(learning_rates),
                       num_masks))

valid_loss  = np.zeros((len(activations),
                       len(small_filters),
                       len(big_filters),
                       len(epochs),
                       len(learning_rates),
                       num_masks))

valid_accu = np.zeros((len(activations),
                       len(small_filters),
                       len(big_filters),
                       len(epochs),
                       len(learning_rates),
                       num_masks))

test_loss  = np.zeros((len(activations),
                       len(small_filters),
                       len(big_filters),
                       len(epochs),
                       len(learning_rates),
                       num_masks))

test_accu = np.zeros((len(activations),
                       len(small_filters),
                       len(big_filters),
                       len(epochs),
                       len(learning_rates),
                       num_masks))

for i in range(num_masks):
    print(i)
    mask_selector = mask_selectors[:i+1]
    cnn = ROI_CNN(mask_selector)
    cnn.get_data(balanced=1, tra_val_split=tra_val_split, use_validation=True)
    cnn.data_augmentation({'rotation':5})
    cnn.set_tf_datasets(batch_size=batch_size)

    for j in range(len(activations)):
        activation = activations[j]
        for k in range(len(small_filters)):
            small_filter = small_filters[k]
            for l in range(len(big_filters)):
                big_filter = big_filters[l]
                for m in range(len(epochs)):
                    epoch = epochs[m]
                    for n in range(len(learning_rates)):
                        lr = learning_rates[n]

                        cnn.build_model(small_filter, big_filter)
                        tl, ta, vl, va = cnn.run(lr=lr, epochs=epoch)

                        train_loss[j, k, l, m, n, i] = tl
                        train_accu[j, k, l, m, n, i] = ta
                        valid_loss[j, k, l, m, n, i] = vl
                        valid_accu[j, k, l, m, n, i] = va

                        t_loss, t_accuracy = cnn.test()
                        test_loss[j, k, l, m, n, i] = t_loss
                        test_accu[j, k, l, m, n, i] = t_accuracy

train_loss_final = np.zeros((num_masks))
train_accu_final = np.zeros((num_masks))
valid_loss_final = np.zeros((num_masks))
valid_accu_final = np.zeros((num_masks))

test_loss_final = np.zeros((num_masks))
test_accu_final = np.zeros((num_masks))

best_inds_collect = []
for i in range(num_masks):
    # Choose result from above with best validation accuracy
    best_inds = np.unravel_index(np.argmax(valid_accu[..., i], axis=None), valid_accu[..., i].shape)
    best_inds_collect.append(best_inds)
    print(best_inds)

    # Get test results
    mask_selector = mask_selectors[:i+1]
    j, k, l, m, n = best_inds
    activation = activations[j]
    small_filter = small_filters[k]
    big_filter = big_filters[l]
    epoch = epochs[m]
    lr = learning_rates[n]

    #cnn = ROI_CNN(mask_selector)
    #cnn.get_data(balanced=1, tra_val_split=tra_val_split, use_validation=True)
    #cnn.data_augmentation({'rotation':5})
    #cnn.set_tf_datasets(batch_size=batch_size)

    #cnn.build_model(small_filter, big_filter)
    #tl, ta, vl, va = cnn.run(lr=lr, epochs=epoch)

    train_loss_final[i] = train_loss[j, k, l, m, n, i]
    train_accu_final[i] = train_accu[j, k, l, m, n, i]
    valid_loss_final[i] = valid_loss[j, k, l, m, n, i]
    valid_accu_final[i] = valid_accu[j, k, l, m, n, i]

    #t_loss, t_accuracy = cnn.test()
    test_loss_final[i] = test_loss[j, k, l, m, n, i]
    test_accu_final[i] = test_accu[j, k, l, m, n, i]

f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(np.arange(1, num_masks+1), train_accu_final, '-o')
ax1.plot(np.arange(1, num_masks+1), valid_accu_final, '-o')
ax1.plot(np.arange(1, num_masks+1), test_accu_final, '-o')
ax1.set_xlabel('Number of ROIs')
ax1.set_ylabel('Accuracy')
ax1.set_ylim([0.5, 1])
ax1.legend(('training', 'validation', 'test'))
#ax1.legend(('training', 'test'))

plt.savefig(SAVE_DIR+'ROI_CNN_results_accuracy.pdf')

f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.plot(np.arange(1, num_masks+1), train_loss_final, '-o')
ax2.plot(np.arange(1, num_masks+1), valid_loss_final, '-o')
ax2.plot(np.arange(1, num_masks+1), test_loss_final, '-o')
ax2.set_xlabel('Number of ROIs')
ax2.set_ylabel('Loss')
ax2.legend(('training', 'validation', 'test'))
#ax2.legend(('training', 'test'))
    
plt.savefig(SAVE_DIR+'ROI_CNN_results_loss.pdf')

save_dict = {'train_loss_final': train_loss_final,
             'valid_loss_final': valid_loss_final,
             'train_accu_final': train_accu_final,
             'valid_accu_final': valid_accu_final,
             'test_accu_final': test_accu_final,
             'test_loss_final': test_loss_final,
             'test_loss': test_loss,
             'test_accu': test_accu,
             'train_loss': train_loss,
             'valid_loss': valid_loss,
             'train_accu': train_accu,
             'valid_accu': valid_accu,
             'best_inds_collect': best_inds_collect}

scipy.io.savemat(SAVE_DIR+'ROI_CNN_results.mat', save_dict)

#plt.show()

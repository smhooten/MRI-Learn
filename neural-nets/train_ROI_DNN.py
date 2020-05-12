import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from MRInet import ROI_DNN

SAVE_DIR = './ROI_DNN_RESULTS2/'

# MOST PREDICTIVE ATLAS MASKS -- FROM SVM RESULTS
mask_selectors = [2,3,28,9,43,10,25,33,13, 
                 12,40,31,32,41,42,21,16,39,
                 44,1,14,11,45,4,22,27,29, 
                 17,30,5,24,6,37,36,38,7, 
                 3,18,35,15,20,34,19,8,26]
mask_selectors.reverse()
# Reverse orders from most to least predictive

# HYPERPARAMETER SELECTIONS
num_masks = 15
activations = ['relu','tanh']
small_denses = [15, 20]
big_denses = [40, 50]
batch_size = 10
tra_val_split = 0.8

epochs = [200, 400]
learning_rates = [1e-5, 1e-3]

# FEATURE SELECTION
train_loss = np.zeros((len(activations),
                       len(small_denses),
                       len(big_denses),
                       len(epochs),
                       len(learning_rates),
                       num_masks))

train_accu = np.zeros((len(activations),
                       len(small_denses),
                       len(big_denses),
                       len(epochs),
                       len(learning_rates),
                       num_masks))

valid_loss  = np.zeros((len(activations),
                       len(small_denses),
                       len(big_denses),
                       len(epochs),
                       len(learning_rates),
                       num_masks))

valid_accu = np.zeros((len(activations),
                       len(small_denses),
                       len(big_denses),
                       len(epochs),
                       len(learning_rates),
                       num_masks))

for i in range(num_masks):
    mask_selector = mask_selectors[:i+1]
    dnn = ROI_DNN(mask_selector)
    dnn.get_data(balanced=1, batch_size=batch_size, tra_val_split=tra_val_split, use_validation=True)

    for j in range(len(activations)):
        activation = activations[j]
        for k in range(len(small_denses)):
            small_dense = small_denses[k]
            for l in range(len(big_denses)):
                big_dense = big_denses[l]
                for m in range(len(epochs)):
                    epoch = epochs[m]
                    for n in range(len(learning_rates)):
                        lr = learning_rates[n]

                        dnn.build_model(small_dense, big_dense, activation=activation)
                        tl, ta, vl, va = dnn.run(lr=lr, epochs=epoch)

                        train_loss[j, k, l, m, n, i] = tl
                        train_accu[j, k, l, m, n, i] = ta
                        valid_loss[j, k, l, m, n, i] = vl
                        valid_accu[j, k, l, m, n, i] = va

train_loss_final = np.zeros((num_masks))
train_accu_final = np.zeros((num_masks))
valid_loss_final = np.zeros((num_masks))
valid_accu_final = np.zeros((num_masks))

test_loss = np.zeros((num_masks))
test_accu = np.zeros((num_masks))

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
    small_dense = small_denses[k]
    big_dense = big_denses[l]
    epoch = epochs[m]
    lr = learning_rates[n]

    dnn = ROI_DNN(mask_selector)
    dnn.get_data(balanced=1, batch_size=batch_size, tra_val_split=tra_val_split, use_validation=False)

    dnn.build_model(small_dense, big_dense, activation=activation)
    tl, ta, vl, va = dnn.run(lr=lr, epochs=epoch)

    train_loss_final[i] = tl
    train_accu_final[i] = ta
    valid_loss_final[i] = vl
    valid_accu_final[i] = va

    t_loss, t_accuracy = dnn.test()
    test_loss[i] = t_loss
    test_accu[i] = t_accuracy

f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(train_accu_final, '-o')
#ax1.plot(valid_accu_final, '-o')
ax1.plot(test_accu, '-o')
ax1.set_xlabel('Number of ROIs')
ax1.set_ylabel('Accuracy')
ax1.set_ylim([0.5, 1])
#ax1.legend(('training', 'validation', 'test'))
ax1.legend(('training', 'test'))

plt.savefig(SAVE_DIR+'ROI_DNN_results_accuracy.pdf')

f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.plot(train_loss_final, '-o')
#ax2.plot(valid_loss_final, '-o')
ax2.plot(test_loss, '-o')
ax2.set_xlabel('Number of ROIs')
ax2.set_ylabel('Loss')
#ax2.legend(('training', 'validation', 'test'))
ax2.legend(('training','test'))
    
plt.savefig(SAVE_DIR+'ROI_DNN_results_loss.pdf')

save_dict = {'train_loss_final': train_loss_final,
             'valid_loss_final': valid_loss_final,
             'test_loss': test_loss,
             'train_accu_final': train_accu_final,
             'valid_accu_final': valid_accu_final,
             'test_accu': test_accu,
             'train_loss': train_loss,
             'valid_loss': valid_loss,
             'train_accu': train_accu,
             'valid_accu': valid_accu,
             'best_inds_collect': best_inds_collect}

scipy.io.savemat(SAVE_DIR+'ROI_DNN_results.mat', save_dict)

#plt.show()

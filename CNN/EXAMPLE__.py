import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
import numpy as np
#from deep_sourcecode.dataset_pneumonia.chest_xray.chest_xray import load_data
from seul_prac.pneumonia_dataload import x_train_f, t_train_f, x_valid_f, t_valid_f,x_test_f,t_test_f
from seul_prac.deep_convnet_ex import DeepConvNet
from seul_prac.trainer_seul import Trainer
import time

# Load data using provided load_data function
#data_path = '/home/s2020102663/deep_sourcecode/dataset_pneumonia/chest_xray/chest_xray/'
#train_path = data_path + 'train/'
#valid_path = data_path + 'val/'
#test_path = data_path + 'test/'
## if __name__:'__main__'이걸로 결과 그냥 가져오는 코드 구현

x_train,t_train = x_train_f, t_train_f
x_valid, t_valid = x_valid_f, t_valid_f
x_test, t_test = x_test_f, t_test_f

# Initialize the model
network = DeepConvNet()

#start_time = time.time()
# Train the model
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000,verbose=True)
trainer.train()

# Plot the loss and accuracy graphs
train_loss_list, train_acc_list = trainer.train_loss_list, trainer.train_acc_list
test_acc_list = trainer.test_acc_list
epochs_list = trainer.epochs_list
print(train_acc_list)
print(test_acc_list)
print(np.array(train_loss_list).shape)
print(epochs_list)
#end_time = time.time()
#excution_time = end_time - start_time
#print(excution_time)

#markers = {'train': 'o', 'test': 's'}
#x = np.arange(len(train_loss_list))
train_acc_Arr = np.array(train_acc_list)
test_acc_Arr = np.array(test_acc_list)
#plt.figure(figsize=(12, 4))
#plt.plot(train_loss_list,'.-','g')
plt.plot(epochs_list, train_acc_Arr, '.-',color = 'r')
plt.plot(epochs_list, test_acc_Arr, '.-',color ='b')
plt.show()
#plt.subplot(1, 2, 1)
#plt.plot(x, train_loss_list, marker='o', label='train', markevery=2)
#plt.plot(x, test_loss_list, marker='s', label='test', markevery=2)
#plt.xlabel("epochs")
#plt.ylabel("loss")
#plt.legend(loc='upper right')

#plt.subplot(1, 2, 2)
#plt.plot(x, train_acc_list, marker='o', label='train', markevery=2)
#plt.plot(x, test_acc_list, marker='s', label='test', markevery=2)
#plt.xlabel("epochs")
#plt.ylabel("accuracy")
#plt.legend(loc='upper right')

#plt.tight_layout()
#plt.show()

# Save the trained model parameters
network.save_params("trained_pneumonia_model.pkl")
print("Saved Network Parameters!")


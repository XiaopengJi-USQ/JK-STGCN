from model.JKSTGCN import build_JKSTGCN
from Utils import *

print(128 * '#')
print('Start to evaluate JKSTGCN.')

# Read config files
utils = Utils()
# Train.json in config folder
para_train = utils.ReadConfig('Train')
# training parameters
channels = int(para_train["channels"])
fold = int(para_train["fold"])
context = int(para_train["context"])
num_epochs = int(para_train["epoch"])
batch_size = int(para_train["batch_size"])
optimizer = para_train["optimizer"]
learn_rate = float(para_train["learn_rate"])
lr_decay = float(para_train["lr_decay"])

# model parameters
# JKSTGCN.json in config folder
para_model =utils.ReadConfig('JKSTGCN')
dense_size = np.array(para_model["Globaldense"])
GLalpha = float(para_model["GLalpha"])
num_of_chev_filters = int(para_model["cheb_filters"])
num_of_time_filters = int(para_model["time_filters"])
time_conv_strides = int(para_model["time_conv_strides"])
time_conv_kernel = int(para_model["time_conv_kernel"])
cheb_k = int(para_model["cheb_k"])
l1 = float(para_model["l1"])
l2 = float(para_model["l2"])
dropout = float(para_model["dropout"])


## path to read data, features and output
Path={"feature":para_train["path_feature"],
        "data":para_train["path_preprocessed_data"],
      "output":para_train["path_output"],
}


# Read data
ReadList = np.load(Path['data'], allow_pickle=True)
Fold_Num = ReadList['Fold_len']

print("Read data successfully")
Fold_Num_c = Fold_Num + 1 - context
print('Number of samples: ', np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')

# optimizer（opt）
opt = keras.optimizers.Adam(lr=learn_rate, decay=lr_decay)

# set l1、l2（regularizer）
if l1 != 0 and l2 != 0:
    regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)
elif l1 != 0 and l2 == 0:
    regularizer = keras.regularizers.l1(l1)
elif l1 == 0 and l2 != 0:
    regularizer = keras.regularizers.l2(l2)
else:
    regularizer = None


# k-fold cross validation
all_scores = []

for i in range(0,10):
    print(128 * '_')
    print('Fold #', i)

    # get i th-fold feature and label

    Features = np.load(Path['feature'] + 'Feature_' + str(i) + '.npz', allow_pickle=True)
    val_feature = Features['val_feature']
    val_targets = Features['val_targets']

    ## using sliding window to add context
    print('Feature', val_feature.shape)
    val_feature, val_targets = AddContext_SingleSub(val_feature, val_targets, context)

    sample_shape = (val_feature.shape[1:])
    print('Feature with context:', val_feature.shape)

    sample_shape = (val_feature.shape[1:])

    model = build_JKSTGCN(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides, time_conv_kernel,
                          sample_shape, dense_size, opt, regularizer, dropout)
    # Evaluate
    # Load weights of best performance
    model.load_weights(Path['output'] + 'Best_model_' + str(i) + '.h5')
    val_mse, val_acc = model.evaluate(val_feature, val_targets, verbose=0)

    # Predict
    predicts = model.predict(val_feature)
    print('Evaluate', val_acc)
    all_scores.append(val_acc)
    AllPred_temp = np.argmax(predicts, axis=1)
    AllTrue_temp = np.argmax(val_targets, axis=1)

    if i == 0:
        AllPred = AllPred_temp
        AllTrue = AllTrue_temp
    else:
        AllPred = np.concatenate((AllPred, AllPred_temp))
        AllTrue = np.concatenate((AllTrue, AllTrue_temp))
    # Fold finish
    print(128 * '_')
    del model,  val_feature, val_targets
# # 4. Final results


# print acc of each fold
print(128 * '=')
print("All folds' acc: ", all_scores)
print("Average acc of each fold: ", np.mean(all_scores))

# Print score to console
print(128 * '=')

PrintScore(AllTrue, AllPred,all_scores,)
# Print score to Result.txt file
PrintScore(AllTrue, AllPred,all_scores,savePath='./')

# Print confusion matrix and save
ConfusionMatrix(AllTrue, AllPred, classes=['W', 'N1', 'N2', 'N3', 'REM'], savePath='./')

print('End of evaluating JKSTGCN.')
print(128 * '#')
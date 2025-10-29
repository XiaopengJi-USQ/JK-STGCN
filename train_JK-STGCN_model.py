import tensorflow as tf
from Utils import *
from model.JKSTGCN import build_JKSTGCN
import time

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


# Create save pathand copy .config to it
if not os.path.exists(Path['feature']):
    os.makedirs(Path['feature'])



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
    train_feature = Features['train_feature']
    val_feature = Features['val_feature']
    train_targets = Features['train_targets']
    val_targets = Features['val_targets']


    ## Use the feature to train JKSTGCN

    print('Feature', train_feature.shape, val_feature.shape)
    train_feature, train_targets = AddContext_MultiSub(train_feature, train_targets,
                                                       np.delete(Fold_Num.copy(), i), context, i)
    val_feature, val_targets = AddContext_SingleSub(val_feature, val_targets, context)


    input_shape = (val_feature.shape[1:])


    print('Feature with context:', train_feature.shape, val_feature.shape)
    model = build_JKSTGCN(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides, time_conv_kernel,
                          input_shape, dense_size, opt, regularizer, dropout)
    if i==0:
        model.summary()
    # train
    history = model.fit(
        x=train_feature,
        y=train_targets,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(val_feature, val_targets),
        verbose=2,
        callbacks=[
            keras.callbacks.ModelCheckpoint(Path['output'] + 'Best_model_' + str(i) + '.h5',
                                                   monitor='val_acc',
                                                   verbose=0,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   period=1)])

    # save the final model
    model.save(Path['output'] + 'JKSTGCN_Final_' + str(i) + '.h5')
    print(128 * '_')
    del model, train_feature, train_targets, val_feature, val_targets

saveFile = open(Path['output'] + "Result_JKSTGCN.txt", 'a+')
print(history.history, file=saveFile)
saveFile.close()

print(128 * '_')
print('End of training JKSTGCN.')


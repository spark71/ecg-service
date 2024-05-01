import data.ptbxl.utils as utils

sampling_frequency=100
datafolder=r'C:\Users\redmi\PycharmProjects\ecg-tool-api\data\ptbxl'
outputfolder=r'C:\Users\redmi\PycharmProjects\ecg-tool-api\data\ptbxl\output'


# datafolder
task='superdiagnostic'

# Load PTB-XL data
data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)

#
# Preprocess label data
labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
# Select relevant data and convert to one-hot
data, labels, Y, _ = utils.select_data(data, labels, task, min_samples=0, outputfolder=outputfolder)# labels.to_csv(r'C:\Users\redmi\PycharmProjects\ptbxl_models\data\ptbxl_labels.csv')



labels.to_csv(r'C:\Users\redmi\PycharmProjects\ecg-tool-api\data\ptbxl\labels.csv', index=False)
# 1-9 for training
X_train = data[labels.strat_fold < 10]
y_train = Y[labels.strat_fold < 10]
# 10 for validation
X_val = data[labels.strat_fold == 10]
y_val = Y[labels.strat_fold == 10]

num_classes = 5         # <=== number of classes in the finetuning dataset
input_shape = [1000,12] # <=== shape of samples, [None, 12] in case of different lengths

X_train.shape, y_train.shape, X_val.shape, y_val.shape
# ((19267, 1000, 12), (19267, 5), (2163, 1000, 12), (2163, 5))


# ------------------------------------------------------
%%
def signals_to_npy_folder( sr, save_folder):
    w = os.walk(fr'C:\Users\redmi\PycharmProjects\ecg-tool-api\data\ptbxl\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\records{sr}')
    # path = r'C:\Users\redmi\PycharmProjects\ptbxl_models\data\ptbxl\records100'
    ecgs = []
    for (dirpath, dirnames, filenames) in w:
        samples = list(map(lambda x: x.split('.')[0], filenames))
        for sample in tqdm(samples):
            num_sample = sample.split('_')[0]
            # print(type(num_sample))
            ecg = wfdb.rdsamp(f'{dirpath}/{sample}')[0]
            # print(num_sample)
            np.save(fr'{save_folder}\{num_sample}.npy', ecg)
            ecgs.append(ecg)
        print(dirpath)
    print(ecgs)
    os.listdir(path)
%%
# signals_to_npy_folder(100, r'C:\Users\redmi\PycharmProjects\ecg-tool-api\data\ptbxl\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\npy_signals100')
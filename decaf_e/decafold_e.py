from decaf_e.model import ModelBuilder
from decaf_e.train import train_and_save_model, k_fold
from decaf_e.visualize_weights import compare_class_attention
from decaf_e.utils import create_results_folder, load_pkl_and_report_shape

epochs = 30
batch_size = 4
dense_layers = 200
dropout_rate = 0.1
data_pos = 0
label_pos = 1

trial = 'pi3k_wt'
data_path = f'../af2_datasets/{trial}'
save_path = f'../results/{trial}'
save_path_model = f'{save_path}/models/ep_{epochs}_ba_{batch_size}_dl_{dense_layers}_dr_{dropout_rate}.h5'

create_results_folder(f"../results/{trial}")
create_results_folder(f"../results/{trial}/plots")
create_results_folder(f"../results/{trial}/csv")
create_results_folder(f"../results/{trial}/sequences")
create_results_folder(f"../results/{trial}/models")

seq_len = load_pkl_and_report_shape(data_path)

msa_model = ModelBuilder(seq_len,
                         dense_layers,
                         dropout_rate).create_model()

# k_fold(msa_model,
#        data_path,
#        save_path,
#        batch_size,
#        epochs,
#        data_pos,
#        label_pos)
#
#
train_and_save_model(msa_model,
                     data_path,
                     save_path_model,
                     batch_size,
                     epochs,
                     data_pos,
                     label_pos)


compare_class_attention(save_path_model,
                      save_path,
                      data_path,
                      data_pos,
                      label_pos,
                      trial)

from decaf_e.cluster_and_label_data import label_data

unlabeled_data_pat = '../unlabeled_datasets/*pkl'

ref1_pos_in_tuple = 3
ref2_pos_in_tuple = 4
rmsd1_thresh = 6
rmsd2_thresh = 1.25
trial = 'pi3k_wt'

label_data(unlabeled_data_pat,
           ref1_pos_in_tuple,
           ref2_pos_in_tuple,
           rmsd1_thresh,
           rmsd2_thresh,
           trial)
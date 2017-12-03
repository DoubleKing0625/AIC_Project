import pandas as pd



def get_train_set():
	data = pd.read_csv('./data/bw_sample.csv')
	cols = ["inf_2", "inf_7", "inf_15", "inf_30", "inf_50", "sup_50", "mean_jx", "min_jx", "max_jx", "nb_occ", "std", "client_id", "mode_nb", "mode_jx"]
	data.columns = cols
	
	return data
	
def get_X(data, useful_cols = ["inf_2", "inf_7", "inf_15", "inf_30", "inf_50", "sup_50", "nb_occ"]):
	X = data[useful_cols]
	for c in [col for col in X.columns if 'nb_occ' not in col]:
		X['prop_' + c] = X[c]/ X['nb_occ']
		del X[c]
	cols = ['prop_inf_2', 'prop_inf_7', 'prop_inf_15', 'prop_inf_30', 'prop_inf_50', 'prop_sup_50']
	X = X[cols]
	
	return X
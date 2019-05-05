import numpy as np


path_ = "wgan_ln/en_en/"
file_name = "vectors-tt.txt"
save_name = "top10k_tt"

# path_ = "wiki_vec/"
# file_name = "wiki.en.vec"
# save_name = "top10k_en_ori"


#===================== read and clearn a vec file =======================

num_line = 0
with open(path_ + file_name, "r") as all_lines:
	arr_lines = []
	print("...start to save")
	for line_index, line in enumerate(all_lines):
		if line_index <= 10000:
			arr_lines.append(line)



arr_lines = np.array(arr_lines)
data_ = arr_lines.reshape(len(arr_lines),1)[1:]
arr_save = np.array([x[0].split() for x in data_])[:,1:]



np.save(path_ + save_name, arr_save)


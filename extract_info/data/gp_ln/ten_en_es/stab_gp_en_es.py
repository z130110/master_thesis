import os

#grad_lambda_list = [0.1, 0.5, 2, 20, 50, 100]
#lr_list = ["adam,lr=0.00001", "adam,lr=0.00005", "adam,lr=0.0001", "adam,lr=0.0005", "adam,lr=0.001", "adam,lr=0.01"]

#random_seeds = [1,2,3,4,5,6,7,8,9,10]
random_seeds = [9,10]

layer_norm = True
number_epochs = 50

lr = "adam,lr=0.0005"
src_lang = "en"
tgt_lang = "es"
norm_ = "center"


print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " + src_lang +"_" + tgt_lang + " ten runs for wgan stability test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

counter = 0
for seed in random_seeds:
	counter += 1
	print("============================================================================================================================================================")
	print("This is the %.dth run for stability test " %(counter))
	#print("The hyperparameters for this run is weight clip = {0}, leanring rate {1} ".format(lr,w_clip))
	print("============================================================================================================================================================")
	os.system(("python3  ../unsupervised.py --layer_norm {0} --map_optimizer {1} --n_epochs " + str(number_epochs) + \
		" --dis_optimizer {2} --seed {3} --normalize_embeddings {4} --src_lang " + src_lang + " --tgt_lang " + tgt_lang + " --src_emb ../../data/wiki." + src_lang + ".vec --tgt_emb ../../data/wiki." + tgt_lang + ".vec ")\
		.format(layer_norm,lr,lr, seed, norm_))

















import numpy as np
import matplotlib.pyplot as plt


class logger(object):
    def __init__(self, path, line_interval):
        self.path = path
        self.line_interval = line_interval  # line from start to the end for single run or whole log file
        self.text_lines, self.num_lines = self.load_file()
        self.d_loss_array = self.extract_d()
        self.g_loss_array = self.extract_g()
        self.num_epochs = round(self.d_loss_array.shape[0] / 250)
        self.knn_cos_dic = self.extract_csls()
        self.best_res = self.best_res()

    def load_file(self):
        with open(self.path, "r") as ins:
            text_lines = []
            for num, line in enumerate(ins):
                if self.line_interval[0] < num < self.line_interval[1]:
                    text_lines.append(line)
        return text_lines, len(text_lines)

    def extract_d(self):
        loss_d_list = []
        for i in range(self.num_lines):
            split_line = self.text_lines[i].split()
            if  "Discriminator" not in split_line and "loss:" not in split_line or "Message:" in split_line:   
                continue
            if "day," in split_line or "days," in split_line:
                temp_loss = split_line[13].lstrip("-")
                temp_loss = temp_loss.replace(".","",1).replace(",","")
                if temp_loss.isdigit():
                    final_loss_d = split_line[13].replace(",","")
                    loss_d = float(final_loss_d)
                    loss_d_list.append(loss_d)
            else:    
                temp_loss = split_line[11].lstrip("-")
                temp_loss = temp_loss.replace(".","",1).replace(",","")
                if temp_loss.isdigit():
                    final_loss_d = split_line[11].replace(",","")
                    loss_d = float(final_loss_d)
                    loss_d_list.append(loss_d)
        return np.array(loss_d_list)

    def extract_g(self):
        loss_g_list = []
        for i in range(self.num_lines):
            split_line = self.text_lines[i].split()
            if "G_loss:" not in split_line or "Message:" in split_line:   
                continue
            if "day," in split_line or "days," in split_line:
                temp_loss = split_line[15].lstrip("-")
                temp_loss = temp_loss.replace(".","",1).replace(",","")
                if temp_loss.isdigit():
                    final_loss_g = split_line[15].replace(",","")
                    loss_g = float(final_loss_g)
                    loss_g_list.append(loss_g)
            else:        
                temp_loss = split_line[13].lstrip("-")
                temp_loss = temp_loss.replace(".","",1).replace(",","")
                if temp_loss.isdigit():
                    final_loss_g = split_line[13].replace(",","")
                    loss_g = float(final_loss_g)
                    loss_g_list.append(loss_g)
        return np.array(loss_g_list)

    def extract_csls(self):
        counter_csls = 0
        counter_nn = 0
        knn_n_to_precision = {}
        csls_1_list = []
        csls_5_list = []
        csls_10_list = []
        nn_1_list = []
        nn_5_list = []
        nn_10_list = []
        cosine_nn_list = []
        cosine_csls_list = []
        n_ = 0
        for line in self.text_lines:
            split_line = line.split()
            if "csls_knn_10" in split_line and "Precision" in split_line and "source" in split_line and "Message:" not in split_line:
                csls_value = float(split_line[-1]) / 100
                residual = counter_csls % 3
                if residual == 0:
                    csls_1_list.append(csls_value)
                elif residual == 1:
                    csls_5_list.append(csls_value)
                elif residual == 2:
                    csls_10_list.append(csls_value)
                counter_csls += 1

            if "nn" in split_line and "at" in split_line and "Precision" in split_line and "source" in split_line and "Message:" not in split_line:
                nn_value = float(split_line[-1]) / 100
                residual = counter_nn % 3
                
                if residual == 0:
                    nn_1_list.append(nn_value)
                elif residual == 1:
                    nn_5_list.append(nn_value)
                elif residual == 2:
                    nn_10_list.append(nn_value)
                counter_nn += 1

            if "Mean" in split_line and "(nn" in split_line and "max" in split_line and "Message:" not in split_line:
                cosine_nn = float(split_line[-1]) 
                cosine_nn_list.append(cosine_nn)
                
            if "Mean" in split_line and "cosine" in split_line and "(csls_knn_10" in split_line and "Message:" not in split_line:
                cosine_csls = float(split_line[-1]) 
                cosine_csls_list.append(cosine_csls)

        knn_n_to_precision["csls@1"] = csls_1_list
        knn_n_to_precision["csls@5"] = csls_5_list 
        knn_n_to_precision["csls@10"] = csls_10_list

        knn_n_to_precision["nn@1"] = nn_1_list
        knn_n_to_precision["nn@5"] = nn_5_list 
        knn_n_to_precision["nn@10"] = nn_10_list

        knn_n_to_precision["cos_nn"] = cosine_nn_list
        knn_n_to_precision["cos_csls"] = cosine_csls_list
        return knn_n_to_precision

    def extract_csls_stability(self):
        res_tem = {}
        res = {}
        csls_5_list = []
        csls_10_list = []
        counter_csls = 0
        num_seed = 0
        seed_list = []

        p1_max = []
        p5_max = []
        p10_max = []

        p1_min = []
        p5_min = []
        p10_min = []

        avoid_refiment = 0
        for line in self.text_lines:
            split_line = line.split()
            res_key = "num_run:" + str(num_seed)

            avoid_refiment += 1
            
            if "Starting" in split_line and "adversarial" in split_line and "training" in split_line:
                avoid_refiment = 0

            if "seed:" in split_line:
                seed_list.append(split_line[-1])
                res_key_p1 = "p1_num_seed:" + str(split_line[-1])
                res_key_p5 = "p5_num_seed:" + str(split_line[-1])
                res_key_p10 = "p10_num_seed:" + str(split_line[-1]) 
                res_tem[res_key_p1] = []
                res_tem[res_key_p5] = []
                res_tem[res_key_p10] = []

            if "csls_knn_10" in split_line and "Precision" in split_line and "source" in split_line and "Message:" not in split_line and avoid_refiment < 285:
                csls_value = float(split_line[-1]) 
                residual = counter_csls % 3
                if residual == 0:
                    res_tem[res_key_p1].append(csls_value)
                elif residual == 1:
                    res_tem[res_key_p5].append(csls_value)
                elif residual == 2:
                    res_tem[res_key_p10].append(csls_value)
                counter_csls += 1

        #print(seed_list)
        for seed in seed_list:
            temp_p1_arr = np.array(res_tem["p1_num_seed:" + str(seed)])
            temp_p5_arr = np.array(res_tem["p5_num_seed:" + str(seed)])
            temp_p10_arr = np.array(res_tem["p10_num_seed:" + str(seed)])

            local_max_index = temp_p10_arr.argmax()          
            #local_min_index = temp_p10_arr.argmin()

            # print(temp_p1_arr)
            # print(temp_p5_arr)
            # print(temp_p10_arr)
            # print("========================")
            p1_max.append(temp_p1_arr[local_max_index])
            p5_max.append(temp_p5_arr[local_max_index])
            p10_max.append(temp_p10_arr[local_max_index])

            #p1_min.append(temp_p1_arr[local_min_index])
            #p5_min.append(temp_p5_arr[local_min_index])
            #p10_min.append(temp_p10_arr[local_min_index])

        global_max_index = np.array(p10_max).argmax()
        global_min_index = np.array(p10_max).argmin()

        res["global_p1_max"] = round(p1_max[global_max_index], 1)
        res["global_p1_min"] = round(p1_max[global_min_index],1)
        res["avg_p1"] = round(np.array(p1_max).mean(),1)
        res["std_p1"] = round(np.array(p1_max).std(),1)

        res["global_p5_max"] = round(p5_max[global_max_index],1)
        res["global_p5_min"] = round(p5_max[global_min_index],1)
        res["avg_p5"] = round(np.array(p5_max).mean(),1)
        res["std_p5"] = round(np.array(p5_max).std(),1)

        res["global_p10_max"] = round(p10_max[global_max_index],1)
        res["global_p10_min"] = round(p10_max[global_min_index],1)       
        res["avg_p10"] = round(np.array(p10_max).mean(),1)     
        res["std_p10"] = round(np.array(p10_max).std(),1)

        res["all_max_p1"] = np.array(p1_max).round(1)
        res["all_max_p5"] = np.array(p5_max).round(1)
        res["all_max_p10"] = np.array(p10_max).round(1)
        return res

    # best training results (without refinement)
    def best_res(self):
        res_dic = {}

        max_p10_index_adv = np.array(self.knn_cos_dic["csls@10"][:self.num_epochs]).argmax()

        res_dic["opt_nn@1"] = round(self.knn_cos_dic["nn@1"][:self.num_epochs][max_p10_index_adv]*100,1)
        res_dic["opt_nn@5"] = round(self.knn_cos_dic["nn@5"][:self.num_epochs][max_p10_index_adv]*100,1)
        res_dic["opt_nn@10"] = round(self.knn_cos_dic["nn@10"][:self.num_epochs][max_p10_index_adv]*100,1)
        res_dic["opt_csls@1"] = round(self.knn_cos_dic["csls@1"][:self.num_epochs][max_p10_index_adv]*100,1)
        res_dic["opt_csls@5"] = round(self.knn_cos_dic["csls@5"][:self.num_epochs][max_p10_index_adv]*100,1)
        res_dic["opt_csls@10"] = round(self.knn_cos_dic["csls@10"][:self.num_epochs][max_p10_index_adv]*100,1)
        res_dic["opt_cos_nn"] = round(self.knn_cos_dic["cos_nn"][:self.num_epochs][max_p10_index_adv]*100,1)
        res_dic["opt_cos_csls"] = round(self.knn_cos_dic["cos_csls"][:self.num_epochs][max_p10_index_adv]*100,1)

        max_p10_index_ref = np.array(self.knn_cos_dic["csls@10"][self.num_epochs:]).argmax()
        res_dic["opt_ref_nn@1"] = round(self.knn_cos_dic["nn@1"][self.num_epochs:][max_p10_index_ref]*100,1)
        res_dic["opt_ref_nn@5"] = round(self.knn_cos_dic["nn@5"][self.num_epochs:][max_p10_index_ref]*100,1)
        res_dic["opt_ref_nn@10"] = round(self.knn_cos_dic["nn@10"][self.num_epochs:][max_p10_index_ref]*100,1)
        res_dic["opt_ref_csls@1"] = round(self.knn_cos_dic["csls@1"][self.num_epochs:][max_p10_index_ref]*100,1)
        res_dic["opt_ref_csls@5"] = round(self.knn_cos_dic["csls@5"][self.num_epochs:][max_p10_index_ref]*100,1)
        res_dic["opt_ref_csls@10"] = round(self.knn_cos_dic["csls@10"][self.num_epochs:][max_p10_index_ref]*100,1)
        res_dic["opt_ref_cos_nn"] = round(self.knn_cos_dic["cos_nn"][self.num_epochs:][max_p10_index_ref]*100,1)
        res_dic["opt_ref_csls"] = round(self.knn_cos_dic["cos_csls"][self.num_epochs:][max_p10_index_ref]*100,1)

        return res_dic

    def plot_loss(self, title_dic, save_path = False, save_or_not = False):
        d_x_indexer = np.arange(1, self.d_loss_array.shape[0] + 1)
        g_x_indexer = np.arange(1, self.g_loss_array.shape[0] + 1)
        label_indexer_d = list(range(self.num_epochs + 1))
        label_indexer_g = list(range(self.num_epochs + 1))
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9,5))
        fig.suptitle(title_dic["title_all"], fontsize= 12)

        ax1.plot(d_x_indexer, self.d_loss_array, color = "red")
        ax1.set_xticks(d_x_indexer[::248])
        ax1.set_xticklabels(label_indexer_d, rotation=45)
        ax1.grid(linestyle='dotted')
        ax1.set_title(title_dic["subtitle_d"])
        plt.tight_layout()

        ax2.plot(g_x_indexer, self.g_loss_array, color = "green")
        ax2.grid(linestyle='dotted')
        ax2.set_title(title_dic["subtitle_g"])
        ax2.set_xticks(g_x_indexer[::248])
        ax2.set_xticklabels(label_indexer_g, rotation=45)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        if save_or_not == True and isinstance(save_path, str):
            plt.savefig(save_path, dpi=600)
        plt.show()

    def plot_loss_more_than_30(self, title_dic, save_path = False, save_or_not = False):
        d_x_indexer = np.arange(1, self.d_loss_array.shape[0] + 1)
        g_x_indexer = np.arange(1, self.g_loss_array.shape[0] + 1)
        label_indexer_d = list(range(self.num_epochs + 1))
        label_indexer_g = list(range(self.num_epochs + 1))
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9,5))
        fig.suptitle(title_dic["title_all"], fontsize= 12)
        ax1.plot(d_x_indexer, self.d_loss_array, color = "red")
        #ax1.set_xticks(d_x_indexer[::248*4])
        #ax1.set_xticklabels(label_indexer_d, rotation=45)
        ax1.grid(linestyle='dotted')
        ax1.set_title(title_dic["subtitle_d"])
        plt.tight_layout()

        ax2.plot(g_x_indexer, self.g_loss_array, color = "green")
        ax2.grid(linestyle='dotted')
        ax2.set_title(title_dic["subtitle_g"])
        #ax2.set_xticks(g_x_indexer[::248*4])
        #ax2.set_xticklabels(label_indexer_g, rotation=45)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        if save_or_not == True and isinstance(save_path, str):
            plt.savefig(save_path, dpi=600)
        plt.show()

    def plot_cosine_with_ref(self, title_dic, save_path = False ,save_or_not = False):
        num_refiments = len(self.knn_cos_dic["csls@1"]) - self.num_epochs
        epoch_index = np.arange(1,self.num_epochs + 1)
        refinement_index = np.arange(self.num_epochs + 1, num_refiments + self.num_epochs + 1)
        plt.plot(epoch_index, self.knn_cos_dic["cos_nn"][:self.num_epochs], '-', markersize = 3 , color = "m", label = "cos nn ")
        plt.plot(epoch_index, self.knn_cos_dic["cos_csls"][:self.num_epochs], '-', markersize = 3 , color = (0.01,0.01,0.01), label = "cos csls")
        plt.plot(refinement_index, self.knn_cos_dic["cos_nn"][self.num_epochs:], '-', color = "m", markersize = 3)
        plt.plot(refinement_index, self.knn_cos_dic["cos_csls"][self.num_epochs:], '-', color = "black", markersize = 3)
        plt.grid(linestyle='dotted')
        plt.xlim(-0.5,self.num_epochs + num_refiments + 0.8)
        plt.ylim(-0.05,1.05)
        plt.legend(loc='center right', bbox_to_anchor=(1.35, 0.81))
        plt.gca().set_xlabel("First {0} tranied by {1}, rest improved by refinement".format(self.num_epochs, title_dic["algorithm"]), fontsize = 9 )
        plt.title("CSLS and cosine with refinement, tranied by {0} \n {1}".format(title_dic["algorithm"], title_dic["hyper"]), fontsize = 9)
        if save_or_not == True and isinstance(save_path, str):
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def plot_csls_cosine_with_ref(self, title_dic, save_path = False ,save_or_not = False):
        num_refiments = len(self.knn_cos_dic["csls@1"]) - self.num_epochs
        epoch_index = np.arange(1,self.num_epochs + 1)
        refinement_index = np.arange(self.num_epochs + 1, num_refiments + self.num_epochs + 1)
        plt.plot(epoch_index, self.knn_cos_dic["csls@1"][:self.num_epochs], color = "orange", label = "$knn = 1$")
        plt.plot(epoch_index, self.knn_cos_dic["csls@5"][:self.num_epochs], color = "g", label = "$knn = 5$")
        plt.plot(epoch_index, self.knn_cos_dic["csls@10"][:self.num_epochs], color = "r", label = "$knn = 10$")
        plt.plot(epoch_index, self.knn_cos_dic["cos_nn"][:self.num_epochs], "bo", markersize = 3 ,label = "cos nn ")
        plt.plot(epoch_index, self.knn_cos_dic["cos_csls"][:self.num_epochs], "ro", markersize = 3 ,label = "cos csls")
        plt.plot(refinement_index, self.knn_cos_dic["csls@1"][self.num_epochs:], color = "orange")
        plt.plot(refinement_index, self.knn_cos_dic["csls@5"][self.num_epochs:], color = "g")
        plt.plot(refinement_index, self.knn_cos_dic["csls@10"][self.num_epochs:], color = "r")
        plt.plot(refinement_index, self.knn_cos_dic["cos_nn"][self.num_epochs:], "bo", markersize = 3)
        plt.plot(refinement_index, self.knn_cos_dic["cos_csls"][self.num_epochs:], "ro", markersize = 3)
        plt.grid(linestyle='dotted')
        plt.xlim(-0.5,self.num_epochs + num_refiments + 0.8)
        plt.ylim(-0.05,1.05)
        plt.legend(loc='center right', bbox_to_anchor=(1.35, 0.81))
        plt.gca().set_xlabel("First {0} tranied by {1}, rest improved by refinement".format(self.num_epochs, title_dic["algorithm"]), fontsize = 9 )
        plt.title("CSLS and cosine with refinement, tranied by {0} \n {1}".format(title_dic["algorithm"], title_dic["hyper"]), fontsize = 9)
        if save_or_not == True and isinstance(save_path, str):
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def plot_csls_cosine_without_ref(self, title_dic, save_path = False ,save_or_not = False):
        epoch_index = np.arange(1,self.num_epochs + 1)
        plt.plot(epoch_index, self.knn_cos_dic["csls@1"][:self.num_epochs], color = "orange", label = "$knn = 1$")
        plt.plot(epoch_index, self.knn_cos_dic["csls@5"][:self.num_epochs], color = "g", label = "$knn = 5$")
        plt.plot(epoch_index, self.knn_cos_dic["csls@10"][:self.num_epochs], color = "r", label = "$knn = 10$")
        plt.plot(epoch_index, self.knn_cos_dic["cos_nn"][:self.num_epochs], "bo", markersize = 3 ,label = "cos nn ")
        plt.plot(epoch_index, self.knn_cos_dic["cos_csls"][:self.num_epochs], "ro", markersize = 3 ,label = "cos csls")
        plt.grid(linestyle='dotted')
        plt.xlim(-0.5,self.num_epochs + 0.8)
        plt.ylim(-0.05,1.05)
        plt.legend(loc='center right', bbox_to_anchor=(1.35, 0.81))
        plt.gca().set_xlabel("{0} epochs,  {1} ".format(self.num_epochs, title_dic["algorithm"]), fontsize = 9 )
        plt.title("CSLS and cosine without refinement, tranied by {0} \n {1}".format(title_dic["algorithm"], title_dic["hyper"]), fontsize = 9)
        if save_or_not == True and isinstance(save_path, str):
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def print_csls_cosine_all_runs(self):
        counter_csls = 0
        counter_nn = 0
        knn_n_to_precision = {}
        csls_1_list = []
        csls_5_list = []
        csls_10_list = []
        nn_1_list = []
        nn_5_list = []
        nn_10_list = []
        cosine_nn_list = []
        cosine_csls_list = []
        n_ = 0
        num_runs = 0
        for line in self.text_lines:
            split_line = line.split()

            if "n_refinement:" in split_line:
                num_runs += 1
                num_ref = split_line[-1]
                knn_n_to_precision["Seperator"] = "=" * 30 + " {0}th run result, last {1} are refinement ".format(num_runs, num_ref) + "=" *30 

            if "csls_knn_10" in split_line and "Precision" in split_line and "Message:" not in split_line:
                csls_value = float(split_line[-1]) / 100
                residual = counter_csls % 3
                if residual == 0:
                    csls_1_list.append(csls_value)
                elif residual == 1:
                    csls_5_list.append(csls_value)
                elif residual == 2:
                    csls_10_list.append(csls_value)
                counter_csls += 1

            if "nn" in split_line and "at" in split_line and "Precision" in split_line and "Message:" not in split_line:
                nn_value = float(split_line[-1]) / 100
                residual = counter_nn % 3
                
                if residual == 0:
                    nn_1_list.append(nn_value)
                elif residual == 1:
                    nn_5_list.append(nn_value)
                elif residual == 2:
                    nn_10_list.append(nn_value)
                counter_nn += 1

            if "Mean" in split_line and "(nn" in split_line and "max" in split_line and "Message:" not in split_line:
                cosine_nn = float(split_line[-1]) 
                cosine_nn_list.append(cosine_nn)
                
            if "Mean" in split_line and "cosine" in split_line and "(csls_knn_10" in split_line and "Message:" not in split_line:
                cosine_csls = float(split_line[-1]) 
                cosine_csls_list.append(cosine_csls)

            if "Writing" in split_line and "source" in split_line:
                knn_n_to_precision["csls@1"] = csls_1_list
                knn_n_to_precision["csls@5"] = csls_5_list
                knn_n_to_precision["csls@10"] = csls_10_list

                knn_n_to_precision["nn@1"] = nn_1_list
                knn_n_to_precision["nn@5"] = nn_5_list 
                knn_n_to_precision["nn@10"] = nn_10_list

                knn_n_to_precision["cos_nn"] = cosine_nn_list
                knn_n_to_precision["cos_csls"] = cosine_csls_list
                
                print(knn_n_to_precision["Seperator"]) 
                print("csls@1:",knn_n_to_precision["csls@1"])
                print()
                print("csls@5" ,knn_n_to_precision["csls@5"])
                print()
                print("csls@10",knn_n_to_precision["csls@10"])
                print()
                print("nn@1", knn_n_to_precision["nn@1"])
                print()
                print("nn@5", knn_n_to_precision["nn@5"])
                print()
                print("nn@10", knn_n_to_precision["nn@10"])
                print()
                print("cos_nn", knn_n_to_precision["cos_nn"])
                print()
                print("cos_csls", knn_n_to_precision["cos_csls"])  
                print()
                print()     

                del csls_1_list[:]
                del csls_5_list[:]
                del csls_10_list[:]
                del nn_1_list[:]
                del nn_5_list[:]
                del nn_10_list[:]
                del cosine_nn_list[:]
                del cosine_csls_list[:]        
        #return knn_n_to_precision

def multi_run_std(list_log):
    # input: a list of logger for different seeds
    num_log = len(list_log)
    p5_li = np.array([])
    p10_li = np.array([])
    res = {}
    for i in range(num_log):
        p5_val = list_log["opt_csls@5"]
        p10_val = list_log["opt_csls@10"]
        p5_li = np.append(p5_li, p5_val) 
        p10_li = np.append(p10_li, p10_val)

    res["mean_p5"] = p5_li.mean()
    res["mean_p10"] = p10_li.mean()
    res["std_p5"] = p5_li.std()
    res["std_p10"] = p10_li.std()
    return res









    # def plot_loss(self, title_dic, save_path, save_or_not = False):
    #     d_x_indexer = np.arange(1, self.d_loss_array.shape[0] + 1)
    #     g_x_indexer = np.arange(1, self.g_loss_array.shape[0] + 1)
    #     fig = plt.figure()
    #     fig.suptitle(title_dic["title_all"], fontsize= 12)
    #     plt.subplot(211)
    #     plt.plot(d_x_indexer, self.d_loss_array, color = "red")
    #     ax1.set_xticks([1,2000,4000,10000])
    #     plt.grid()
    #     plt.title(title_dic["subtitle_d"])
    #     plt.subplot(212)
    #     plt.tight_layout()
    #     plt.plot(g_x_indexer, self.g_loss_array, color = "green")
    #     plt.grid()
    #     plt.title(title_dic["subtitle_g"])
    #     fig.tight_layout()
    #     fig.subplots_adjust(top=0.85)
    #     if save_or_not == True:
    #         #print(save_path)
    #         plt.savefig(save_path, dpi=600)
    #     plt.show()



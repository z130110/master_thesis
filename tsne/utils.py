import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA
from sklearn import manifold
import time
from scipy.stats import zscore
import random


def plot_tsne(src_emb, tgt_emb, colors):
    fig, ax = plt.subplots()
    ax.scatter(src_emb[:,0], src_emb[:,1], s=40,c = colors[0], label = "src",
                   alpha=0.3, edgecolors='none')
    ax.scatter(tgt_emb[:,0], tgt_emb[:,1], s=40,c = colors[1], label = "tgt",
                   alpha=0.3, edgecolors='none')
    ax.legend()
    ax.grid(linestyle='dotted')
    ax.axis('off')
    ax.axis('tight')
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.show()


def plot_tsne_three(plot_dic, save_path = "", legend_ = ""):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(plot_dic["src_emb"][:,0], plot_dic["src_emb"][:,1], s=15,c = plot_dic["colors"][0], label = "src(mapped)",
                   alpha=0.4, edgecolors='none')
    ax.scatter(plot_dic["tgt_emb"][:,0], plot_dic["tgt_emb"][:,1], s=15,c = plot_dic["colors"][1], label = "tgt",
                   alpha=0.4, edgecolors='none')
    ax.scatter(plot_dic["src_ori"][:,0], plot_dic["src_ori"][:,1], s=15,c = plot_dic["colors"][2], label = "src_before",
                   alpha=0.4, edgecolors='none')
    
    if legend_:
        ax.legend(borderpad = 0.3, labelspacing = 0.3, markerscale= 2, loc='center right', bbox_to_anchor=(0.27, 0.91))
    ax.grid(linestyle='dotted')
    ax.axis('off')
    ax.axis('tight')
    #plt.title(plot_dic["title"], x = 0.53, y= -0.08)
    plt.xlim(-75, 75)
    plt.ylim(-75, 80)
    #plt.legend(loc='center right', bbox_to_anchor=(0.25, 0.92))
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

def pca_tsne_reduction(file_dic, pca_component = 150, random_seed = 1):
    src_emb = zscore(np.load(file_dic["src_emb_path"]).astype(np.float))
    tgt_emb  = zscore(np.load(file_dic["tgt_emb_path"]).astype(np.float))
    src_ori_emb = zscore(np.load(file_dic["src_ori_path"]).astype(np.float))

    all_emb = np.concatenate((src_emb, tgt_emb, src_ori_emb), axis = 0)
    #all_emb_zscore = zscore(all_emb)

    time_start = time.time()

    # fit_pca = pca.fit(tgt_emb_zscore)
    # fit_pca.transform(all_emb_zscore)

    # pcafit  pcacom.fit(tgt_emb)
    # pcacom.transform*all

    pca_com = PCA(n_components = pca_component)
    pca_fit_tgt = pca_com.fit(tgt_emb)
    #pca_transform_tgt = pca_fit_tgt.transform(tgt_emb)
    pca_transform_all = pca_fit_tgt.transform(all_emb)


    #pca_fit = pca_com.fit_transform(all_emb_zscore)

    print('PCA with {0} components, time elapsed: {1} seconds'.format(pca_component, time.time() - time_start))
    print('Cumulative variance explained by {0} principal components: {1}'.format(pca_component, np.sum(pca_com.explained_variance_ratio_)))

    
    time_start = time.time()
    pca_tsne_transform = manifold.TSNE(random_state= random_seed).fit_transform(pca_transform_all)
    #pca_tsne = manifold.TSNE(random_state= random_seed)
    #pca_tsne_fit = pca_tsne.fit(pca_transform_tgt)
    #pca_tsne_transform = pca_tsne_fit.transform(pca_transform_all)

    print('t-SNE done! time elapsed: {} seconds'.format(time.time()-time_start))
    
    return pca_tsne_transform
    


def random_selection(tsne_res, num_select):
    src_ind = random.sample(range(0,10000), num_select)
    tgt_ind = random.sample(range(10000,20000), num_select)
    src_ori_ind = random.sample(range(20000,30000), num_select)
    src_tsne = tsne_res[src_ind]
    tgt_tsne = tsne_res[tgt_ind]
    src_tsne_ori= tsne_res[src_ori_ind]
    return src_tsne, tgt_tsne, src_tsne_ori





































'''
def load_mnist(path, kind='train'):
    import os
    import gzip

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels



def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)],
        label = palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.legend()
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    
    # txts = []

    # for i in range(num_classes):

    #     # Position of each label at median of data points.

    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)

    return f, ax, sc
'''


import wandb
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

matplotlib.use('Agg')
flatui = ["#9b59b6", "#3498db", "orange"]


def plot_attention_weights(attentions):
    num_layer = len(attentions)
    num_attn= attentions[0].shape[0]
    fig = plt.figure(figsize=(num_attn, num_layer))

    for layer in range(num_layer):
        for head in range(num_attn):
            ax = fig.add_subplot(num_layer, num_attn, (layer * num_attn) + head+1)

            # plot the attention weights
            ax.matshow(attentions[layer][head], cmap='viridis')

            fontdict = {'fontsize': 7}

            ax.set_xlabel('Head {}'.format(head+1), fontdict={'fontsize': 5})
            ax.axis('off')

    plt.tight_layout()

def tsne(x):
    test_features = x.cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=10, n_iter=250) #300)
    tsne_ref = tsne.fit_transform(test_features)
    return tsne_ref


def pca(x):
    test_features = x.cpu().numpy()
    pca = PCA(n_components=2)
    pca.fit(test_features)
    pcs = pca.fit_transform(test_features)
    return pcs


def df_tsne(tsne_ref , label):
    df = pd.DataFrame(tsne_ref, index=tsne_ref[0:,1])
    df['x1'] = tsne_ref[:,0]
    df['x2'] = tsne_ref[:,1]
    df['label'] = label
    return df


def df_pca(pcs, label):
    df = pd.DataFrame(data = pcs, columns = ['x1',  'x2'])
    label = pd.DataFrame(label)
    df = pd.concat([df,label],axis = 1,join='inner', ignore_index=True)
    df = df.loc[:,~df.columns.duplicated()]
    df.columns = ['x1', 'x2', 'label']
    return df


def plot_sns_scatter(df, label):
    # sns.set_palette(flatui)
    sns.scatterplot(x="x1", y="x2", hue='label', data=df, legend=True, \
        palette=sns.color_palette("hls", 10), scatter_kws={"s":50, "alpha":0.5})


def plot_sns_lm(df, label):
    sns.set_palette(flatui)
    sns.lmplot(x="x1", y="x2", data=df, fit_reg=False, legend=True, hue='label', scatter_kws={"s":50, "alpha":0.5})


def plot_plt_scatter(tsne_ref, label):
    f, ax = plt.subplots()
    # sns.set_palette(flatui)
    cmap = sns.color_palette("light:#9b59b6", as_cmap=True) #sns.cubehelix_palette(as_cmap=True)
    points = ax.scatter(tsne_ref[:,0], tsne_ref[:,1], c=label, s=50, cmap=cmap)
    f.colorbar(points)


def plot_dr(x_tsne, x_pca, label, i_plot, name):
    label = label.cpu().numpy()

    def plot_subdr(xdr, drf, plotf, ptitle, wtitle, axis_label=True):
        result = drf(xdr, label)
        plotf(result, label)
        plt.title('{}'.format(ptitle, name), weight='bold').set_fontsize('14')
        if axis_label:
            plt.xlabel('u1', weight='bold').set_fontsize('14')
            plt.ylabel('u2', weight='bold').set_fontsize('14')
        # wandb.log({"{}{}".format(wtitle, name): wandb.Image(plt)})

    need_df = i_plot != 2
    tsnef = df_tsne if need_df else lambda x, y: x
    pcaf = df_pca if need_df else lambda x, y: x
    plotf = [plot_sns_scatter, plot_sns_lm, plot_plt_scatter][i_plot]
    plot_subdr(x_tsne, tsnef, plotf, 't-SNE: ', 'tsne_', False)
    plot_subdr(x_pca, pcaf, plotf, 'PCA: ', 'pca_')


def plot_tsne(features, labels):
    ''' 
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签 
    '''
    import os
    os.makedirs('fig', exist_ok=True)
    print('features的shape:',features.shape)
    print('labels的shape:',labels.shape)
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    class_num = len(np.unique(labels)) #要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    latent = features
    tsne_features = tsne.fit_transform(features)    #将特征使用PCA降维至2维
    print('tsne_features的shape:',tsne_features.shape)
    plt.clf()
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1])   #将对降维的特征进行可视化

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:,0]
    df["comp-2"] = tsne_features[:,1]
    
    unique_labels = sorted(np.unique(labels))
    palette = dict(zip(unique_labels, sns.color_palette("hls", class_num)))

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=palette,legend=False,
                    data=df).set()# title="Bearing data T-SNE projection unsupervised"
    global prediction_ratio
    plt.title(f"t-SNE projection of the {prediction_ratio} ratio data")
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.savefig(f"fig/tsne_scatterplot_mask_{prediction_ratio}.png")
    print("保存图片成功")
    plt.close()


def main():
    global prediction_ratio 
    prediction_ratio_list = [0.2,0.4,0.6,0.8,1.0] #[1.0]#
    for prediction_ratio in prediction_ratio_list:
        data_path = f'/data/data1/zhengchaoyang/infogcn/results/ntu_NTU60_CS_8_{prediction_ratio}/z_values.npz'
        z_data = np.load(data_path)
        mask = np.isin(z_data['y'], np.arange(20))
        plot_tsne(z_data['z'][mask], z_data['y'][mask])

if __name__ == "__main__":
    # Example usage
    main()
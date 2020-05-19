#!/usr/bin/env python3

''' load model data '''
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('jeed1488.keyed_vecors.bin', binary=True)


''' emoji similarity '''
# find out more similarity-related methods within the gensim api documentation - https://radimrehurek.com/gensim/
for e in ['🍻', '🔨', '🏕', '🕵', '🙈', '🌈', '🏰', '🍑', '➕', '😉', '🚌']:
    print('most similar to', e, 'are:', model.most_similar(e))



''' tSNE visualiuation '''
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# different parameters will result in a different projection.
tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=500, random_state=42)
# the fitting may take some time.
tsne_results = pd.DataFrame(tsne.fit_transform(model.vectors), columns=['tsne-2d-one', 'tsne-2d-two'])
# get the corresponding emoji groups in order
emojis_groups = pd.read_json('jeed1488_emoji_groups.json.gz')
tsne_results['group'] = [emojis_groups[emojis_groups['emoji'] == emoji]['group'].iloc[0] for emoji in model.vocab]

f = plt.figure(figsize=(16,10))
ax = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="group",
    palette=sns.color_palette("Paired", len(set(emojis_groups['group']))),
    data=tsne_results,
    legend='full',
    alpha=1
)
plt.savefig('''tsne.png''', dpi=600)
plt.show()
plt.close()

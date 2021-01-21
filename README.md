# MGPred
we develop a novel prediction model for drug side effect frequencies, using a multi-view graph convolutional model to integrate three different types of features, including similarity, association distribution, and word embedding.



# Requirements
* python == 3.6
* pytorch == 1.6
* Numpy == 1.16.2
* scikit-learn == 0.21.3


# Files:

1.data

This folder contains 5 input files needed by our model.

Drug_word2vec.pkl: The word embedding matrix of drugs. We use Mol2vec model to learn the word embedding of drugs. Mol2vec can learn vector representations of molecular substrctures pointing to similar directions of chemically related substructures. Each row of the matrix represents the word vector encoding of a drug.

glove_wordEmbedding.pkl: The word embedding matrix of side effects. We use the 300-dimensional Global Vectors (GloVe) trained on the Wikipedia dataset to represent the information of side effects. Each row of the matrix represents the word vector encoding of a side effect.

effect_side_semantic.pkl: The semantic similarity matrix of side effects. We download side effect descriptors from Adverse Drug Reaction Classification System (ADReCS, http://bioinf.xmu.edu.cn/ADReCS/index.jsp), and construct a novel model to calculate the semantic similarity of side effects. Each row of the matrix represents the similarity value between a side effect and all side effects in the benchmark dataset. The range of values is from 0 to 1.

Text_similarity.pkl: The similarity matrix of drugs. The matrix is collected from the file "Chemical_chemical.links.detailed.v5.0.tsv.gz" in STITCH database (http://stitch.embl.de/). Each row of the matrix represents the similarity value between a drug and all drugs in the benchmark dataset. The range of values is from 0 to 1.

drug_side.pkl: In summary, the benchmark dataset contains 37,071 frequency classes that cover 750 drugs and 994 side effects. The matrix has 750 rows and 994 columns to store the known drug-side effect frequency pairs. The element at the corresponding position of the matrix is set to the frequency value, otherwise 0.

If you want to view the value stored in the file, you can run the following command:

```bash
import pickle
import numpy as np
gii = open(‘data’ + '/' + ' drug_side.pkl ', 'rb')
drug_side_effect = pickle.load(gii)
```

2. utils

This folder contains some function files

aggregator.py: for aggregating feature of neighbors.

attention.py: This function can aggregate the features of neighbors

encoder.py: This function can aggregate its own features and the neighbor features.

l0dense.py: Implementation of L0 regularization for the input units of a fully connected layer.

3.Code

Ten_Fold_test.py: This function can test the predictive performance of our model under ten-fold cross-validation.


# Train and test folds
python Ten_Fold_test.py --rawpath /Your path --epochs Your number --batch_size Your number

rawdata_dir: All input data should be placed in the folder of this path. (The data folder we uploaded contains all the required data.)

epochs: Define the maximum number of epochs.

batch_size: Define the number of batch size for training and testing.

All files of Data and Code should be stored in the same folder to run the model.

Example:

```bash
python cross_validation.py --rawdata_dir /data --epochs 100 --batch_size 256
```
# Contact 
If you have any questions or suggestions with the code, please let us know. Contact Haochen Zhao at zhaohaochen@csu.edu.cn

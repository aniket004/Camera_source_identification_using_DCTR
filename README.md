# Camera_source_identification_using_DCTR
CVPRW_2017

1. For Details refer to paper: Aniket Roy, Rajat Subhra Chakraborty, Venkata Udaya Sameer, Ruchira Naskar:
"Camera Source Identification Using Discrete Cosine Transform Residue Features and Ensemble Classifier." CVPR Workshops 2017: 1848-1854

2. DCTR features are extracted using DCTR.m . 
refer to ( V. Holub and J. Fridrich, "Low Complexity Features for JPEG Steganalysis Using
Undecimated DCT", IEEE Transactions on Information Forensics and Security,
DCTR feature : http://dde.binghamton.edu/download/feature_extractors/)

3. Train the classifier using Random forest with optimal no of components of PCA: run opt_itr_PCA.m

4. For Adaboost use: adaboost_M2.m

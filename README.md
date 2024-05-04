# Unsupervised Domain Adaptation Within Deep Foundation Latent Spaces

Paper: 
Dmitry Kangin & Plamen Angelov (2024), Unsupervised Domain Adaptation Within Deep Foundation Latent Spaces, ICLR workshop on Mathematical and Empirical Understanding of Foundation Models, 2024

[Paper link](https://openreview.net/forum?id=5z5XyMqZXW&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FWorkshop%2FME-FoMo%2FAuthors%23your-submissions))

![domain_adaptation_scheme](https://github.com/lira-centre/vit_uda/assets/5869514/b0e8bbcd-e557-44d7-9e87-37b6c31341cc)

The methodology scheme: (1) the images from multiple domains (e.g., sketches and real images) are embedded into the feature space and, for each domain, separately clustered using $k$-means. The cluster centroids for one of the domains ('source domain'), shown in bright colour in the figure and referred to as 'prototypes', are provided with labels. (2) Domain adaptation is performed through inter-domain cluster matches with $\ell^2$ or Wasserstein distance. (3) Decision making through nearest-neighbour prototype classifier performs the prediction

The instructions for running the code are given below. 

1. Download DomainNet dataset from here: http://ai.bu.edu/M3SDA/ and put it in the folder ../UnsupervisedDomainAdaptation
2. Run extract_all_features_uda.sh for feature extraction
3. Create log directory, e.g. logs_dinov2_vit_g14 and run run_all.sh
4. The log files in the directory logs_dinov2_vit_g14 will contain performance of the model for L2 and Wasserstein distances
5. To change the backbone model, change line 25 in domain_adaptation.py to select one of the following options for the variable MODEL: 'dinov2_vitg14', 'resnet152', 'vit_h14_in1k'

# Unsupervised Domain Adaptation Within Deep Foundation Latent Spaces

Paper: 
Dmitry Kangin & Plamen Angelov (2024), Unsupervised Domain Adaptation Within Deep Foundation Latent Spaces, ICLR workshop on Mathematical and Empirical Understanding of Foundation Models, 2024

This code is provided for the purposes of reproducibility. The instructions for running the code are given below. 

1. Download DomainNet dataset from here: http://ai.bu.edu/M3SDA/ and put it in the folder ../UnsupervisedDomainAdaptation
2. Run extract_all_features_uda.sh for feature extraction
3. Create log directory, e.g. logs_dinov2_vit_g14 and run run_all.sh
4. The log files in the directory logs_dinov2_vit_g14 will contain performance of the model for L2 and Wasserstein distances
5. To change the backbone model, change line 25 in domain_adaptation.py to select one of the following options for the variable MODEL: 'dinov2_vitg14', 'resnet152', 'vit_h14_in1k'

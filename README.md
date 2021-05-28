# Music-Speech-Classification
Unsupervised Music Speech Classification using Gaussian Mixture Models (EM algorithm)

Problem Description -

A set of training and test examples of music and speech are provided.
http : //www.leap.ee.iisc.ac.in/sriram/teaching/MLSP21/assignments/speechMusicData.tar.gz

Using these examples,
  a) Generate spectrogram features - Use the log magnitude spectrogram as before with a 64 component magnitude FFT (NFFT). In this case, the spectrogram will have
dimension 32 times the number of frames (using 25 ms with a shift of 10 ms).
  b) Train two GMM models with K-means initialization (for each class) separately each with 5-mixture components with diagonal/full covariance respectively on this data. Plot the log-likelihood as a function of the EM iteration.
  c) Classify the test samples using the built classifiers and report the performance in terms of error rate (percentage of mis-classified samples) on the text data.

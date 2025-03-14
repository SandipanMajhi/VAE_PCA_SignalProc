# VAE-PCA Signal Processing

### Ideation ###
Dimensionality Reduction Techniques used : PCA and VAE
Step 1: Take the spectral features and reduce dimensions using PCA or VAE
  Extra Step : For VAE, pre-train on the training set to reconstruction loss and KL divergence loss on Standard Normal Distirbution for latent space
Step 2: After Projection, use a small neural network to train on the regression targets.


### Target Scaling ###
The targets were scaled using a MinMax Scaler so the output layer of the Neural Network has Sigmoid activation which trained the model quite easily. Then we optmize the Binary Cross Entropy Loss function on the output. From our tests, we found that mean squared loss is not a suitable loss function for the labels. We were unable to train it on MSE Loss. 

### Results ###
`Targets scaled to [0,1] using MinMax Scaler`

`Train Split = 0.8`

`Val Split = 0.1`

`Test Split = 0.1`'

`VAE and PCA Latent dimension = 32`

`Regression NN Model =  32 (input features) --(Relu)--> 128 --(Relu)--> 64 --(Relu)--> 8 --(Relu)--> 1 ---> Sigmoid `

| Model                                   | Test Loss  | 
|-----------------------------------------|------------|
| Regression NN + PCA Latent 32 dimension | 0.18186    |
| Regression NN + VAE Latent 32 dimension | 0.24823    |



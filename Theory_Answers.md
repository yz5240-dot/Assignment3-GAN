# Assignment 3: Building Blocks of GAN

Yulin Zhang

## Question 1

For dilation 1, the output size along each spatial dimension is:

O = (I - 1)S - 2P + K + OP

Substituting I = 8, K = 4, S = 2, P = 1, and OP = 1:

O = (8 - 1)(2) - 2(1) + 4 + 1 = 14 - 2 + 4 + 1 = 17.

Therefore, the output feature map is **17 x 17**.

## Question 2

Keeping the other values from Question 1, set S = 3:

O = (8 - 1)(3) - 2(1) + 4 + 1 = 21 - 2 + 4 + 1 = 24.

The output changes from **17 x 17 to 24 x 24**, an increase of 7 in each dimension. In general, increasing the stride by 1 increases each output dimension by I - 1, where I is that dimension's input size.

## Question 3

With dilation equal to 1:

**O = (I - 1)S - 2P + K + OP.**

Apply this formula separately to the height and width. I is input size, K is kernel size, S is stride, P is padding, and OP is output padding.

If dilation D is included, the general formula is:

O = (I - 1)S - 2P + D(K - 1) + OP + 1.

## Question 4

One possible configuration is **kernel size 2 x 2, stride 2, padding 0, output padding 0**, and dilation 1.

O = (16 - 1)(2) - 2(0) + 2 + 0 = 30 + 2 = 32.

Therefore, it upsamples a 16 x 16 feature map to **32 x 32**.

## Question 5

The mini-batch is [6, 8, 10, 6].

Mean:

mu = (6 + 8 + 10 + 6) / 4 = 7.5.

Batch variance (divide by 4):

variance = [(6 - 7.5)^2 + (8 - 7.5)^2 + (10 - 7.5)^2 + (6 - 7.5)^2] / 4

= (2.25 + 0.25 + 6.25 + 2.25) / 4 = 2.75.

Standard deviation = sqrt(2.75) = 1.6583.

Without gamma and beta, each normalized value is (x - mu) / sqrt(variance).

- 6: -1.5 / 1.6583 = -0.9045
- 8: 0.5 / 1.6583 = 0.3015
- 10: 2.5 / 1.6583 = 1.5076
- 6: -1.5 / 1.6583 = -0.9045

The normalized output is **[-0.9045, 0.3015, 1.5076, -0.9045]**.

This calculation ignores epsilon. Implementations use sqrt(variance + epsilon) for numerical stability; epsilon makes the variance of the normalized batch slightly less than 1. The normalization uses the biased batch variance, not the sample variance with denominator n - 1.

## Question 6

**ReLU(x) = max(0, x)**. Equivalently, ReLU(x) = x for x >= 0 and 0 for x < 0.

**LeakyReLU(x) = x for x >= 0, and alpha*x for x < 0**, where alpha is a small positive constant, such as 0.2 in this assignment.

The key difference is the negative-input slope: ReLU has slope 0, whereas LeakyReLU has slope alpha.

## Question 7

LeakyReLU allows a small gradient to pass through when the input is negative. With ReLU, the gradient is zero in that region, so some neurons may stop learning. LeakyReLU reduces this dying-ReLU problem and can help gradients flow through deep networks. It is often used in GAN discriminators. It does not guarantee stable GAN training or better results in every network.

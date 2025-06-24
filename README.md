---
title: Project Description

---



## Project Description

We want to learn the geometry of morphing bird wings. However, due to limited resources, semi-automatic methods are used to label the raw data. To reduce the amount of labour involved, we wish to model the labels probailistically and bring the labelling process to be fully automated.


## Data Description

These are the two main datasets we will be focused on.

### FullBilateralMarkers.csv

Quoted from Lydia, this our Gold standard labelled data:

- One row is one frame
- Every row must contain exactly 8 markers (no nans)
- All markers are labelled
- Expected labelling error is about 4%

### FullNoLabels.csv

This is the tricky one, it contains everything recorded by the motion capture but unlabelled (Lydia). 


## Main Problem

1. Some (many) of the labels are missing and we have the coordinates for it. It may be because of 
    a. We don't know what it is (without manual methods) and it is indeed a label on the bird
    b. It is a spurious extra label
2. We even may not observe the labels, i.e. no label, no coordinates. 
3. In the Gold standard dataset, we have an order for the labels (e.g. 'left_wingtip', 'right_wingtip', 'left_primary', ...), but it is not necessary the case in the unlabelled dataset, thus also our predictions.


## What Lydia has done to help with this problem

Lydia has applied DMD to the partially missing data. She visualized it with observed data overlayed on top of the DMD output. For a flight sequence, she took the part where the bird is flapping and when the observed data is gliding, she let the DMD output to project the flapping actions.

Pointed out by Ben, the missing parts can be long and thus there could be more than 1 mode in the missing parts. 


## First thoughts in 23/6/25 meeting

A generative approach will be useful since we can easily add new markers later. We model the coordinates first then the markers conditioned on the coordinates.

For the coordinates, let
- $Y(t) = (Y^1(t),...,Y^8(t))$ be the predictions of markers from DMD, or possibly non processed bird,
- $X(t) = (X^1(t),...,X^k(t))$ be the observations where some of them are missing, where $k$ can be not equal to $8$, to cover the situations where 1. we cannot obverse some of the labels, and 2. there are some spurious extra labels.
- $Z(t) = (Z^1(t),...,Z^k(t))$ be the true coordinates of the labels that can be observed. 
- $t \in \mathbb{R}$ be the time of a flight sequence.

Since the DMD output is reasonably close to the real bird, we assume that the true underlying model is $Z(t)|Y(t)$, and $X(t)$ are the realisationis from $Z(t)$. We start modelling the likelihood naively and generalise it procedurally. 

Firsly, we assume $k=8$ and model each marker independently,

$$Z^i(t) \sim N(Y^i(t), \sigma), \text{ for } i = 1,...,8$$

This is already a multivariate normal since we are using 3D coordinates. I assume the xyz coordinates are independent, since we cannot tell the coordinates of xy given z for example.

However, it is apparent that the coordinates of different markers are correlated, e.g. if the left wing tip is high, the right wing tip is probably also high, like in a flapping action. So
$$
Z(t) \sim N(Y(t), \Sigma)
$$

Nevertheless, we are still assuming 
1. length of $Z(t)$ is the same as $Y(t)$.
2. order of the entries in $Z(t)$ is the same as $Y(t)$.

Taking this in account, let $k$ be the length of $Z(t)$, and $J(t)$ be the vector of markers of $Z(t)$, where entry $i = 1,...,k$ of $J(t)$, $J^i(t) \in \{ 1,2,...,8,*\}$, $*$ represents that the corresponding $Z^i(t)$ is invalid, since it could be a spurious marker or we did not observe all 8 markers.

Since our ultimate goal is to automate the labelling procedure, we are interested in $J(t)|X(t),Y(t)$. In English (for my sanity), we are interested in the labels given the observed coordinates and the DMD predictions at time $t$.
The true posterior is
$$
P(J(t) | Z(t), Y(t)) \propto P(Z(t) | J(t), Y(t)) P(J(t))
$$

We find difficulties in inferring $J(t)$ directly, so instead  Lydia suggested a two-step prediction. Ben followed up with introducing $M(t)$, a length $k$ vector of indicators,
$$
M^i(t) = 
\begin{cases} 
1 & \text{if entry i is a marker} \\
0 & \text{otherwise}
\end{cases}
$$
To clarify, if we know $J(t)$, we know $M(t)$, since indices that are not $*$ in $J(t)$ must be $1$ in $M(t)$, and indices that are $*$ in $J(t)$ must be $0$ in $M(t)$. However, the reverse is not true, since having a $1$ in $M(t)$ does not tell anything about the type of marker in $J(t)$. We only know it is a valid marker. 

Now our parameter of interest becomes the joint $(J(t), M(t))$. The posterior is,
$$
P(J(t), M(t) \mid Z(t), Y(t))
= P(J(t) \mid M(t), Z(t), Y(t)) P(M(t) \mid Z(t), Y(t))
$$
The term $P(J(t) \mid M(t), Z(t), Y(t))$ can be trained by using the Gold standard and $P(M(t) \mid Z(t), Y(t))$ can be trained by cross-referencing the Gold standard (clean) and the unlabelled data (messy). We also mentioned adding artificial noise can be helpful in training this quantity.

I find this two step quantity rather weird, since if we are using a generative approach, we will have two distributions of $Z(t)$, namely $P(Z(t) \mid J(t), Y(t))$ and $P(Z(t) \mid M(t), Y(t))$. Some maths will give
$$
\begin{align}
&\hspace{7mm} P(J(t), M(t) \mid Z(t), Y(t)) \\
&= P(J(t) \mid M(t), Z(t), Y(t)) P(M(t) \mid Z(t), Y(t)) \\
&= P(Z(t) \mid J(t), Y(t)) P(Z(t) \mid M(t), Y(t)) P(M(t)) P(J(t))
\end{align}
$$
Since $M(t)$ is given conditioned on $J(t)$, we will still obtain the problematic $P(Z(t)|J(t),Y(t))$. So in a generative approach, this two step progress is not very helpful?

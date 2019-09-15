# Machine Learning for prediction of molecular properties using a graph convolutional network
This project is my pipeline for generating submissions for the Kaggle "Predicting Molecular Properties" competition. The goal is to predict NMR coupling constants for 2,505,541 atom pairs in the test set, using a training set of 4,658,147 atom pairs.

I started on the contest two weeks before it ended, so time pressure was a factor in my choice of strategy. My submission achieved a score in the top 40% of entries -- ok for a start. My pipeline (shell script, python scripts, training and test datasets, molecular structures) are included. Below I've outlined my strategy, features, the structure of my model(a graph convolutional NN implemented in PyTorch) and how I think this could be improved if I were to try this again.

## Background
NMR is a technique related to MRI used by synthetic chemists to determine the structure of a molecule, like "MRI for molecules". Among the outputs generated by NMR are floating-point values called coupling constants, caused by magnetic interactions between pairs of atoms in molecules. The coupling constant for a specific pair of atoms in a molecule can be predicted using physics, but the calculations involved are computationally quite expensive - <img src="https://latex.codecogs.com/gif.latex?O_t=\text { O(N^4) } t" /> or worse -- which means you don't want to do that unless you absolutely have to.

If coupling constants for specific atom pairs could be accurately predicted directly from the molecular structure, it would be very helpful for building software to automatically determine structure from NMR data (look, no synthetic chemists required!) Molecular structures are graphs where atoms are vertices and edges are bonds. This dataset gives us 3D molecular structures, so for each vertex in each graph we have Cartesian coordinates, as for example:

![figure 0](fig0.png)

I've never worked with NMR data, my background is comp bio / comp chem not synthesis, but I had a little domain knowledge thanks to my background...and after I dug up my old organic chemistry textbook and skimmed a couple chapters I had quite a bit more :). Vollhardt and Schore have various rules-of-thumb for making rough coupling constant predictions. They identify the key factors as these:

1) The electron density (how much negative charge is concentrated on) for the two atoms, which is determined by the type of atom (carbon, oxygen etc.), the atoms bonded to it and the types of bonds connecting them (double, single, etc). Atoms can also be assigned a "partial charge" which approximates how much electron density they have.

2) The bond angles (for coupling constants between two atoms that are separated by one or more neighbors).

3) The electron density on the atoms attached to and connecting the two coupling partners.

I'd worked previously with a Python library called RDKit for handling molecular structures and given my familiarity used it for generating features. Partly due to time limitations, I leaned pretty heavily on domain knowledge for feature engineering.

In addition to the molecular structure, the competition provided some extra features for the training set only. If I were to revisit this, I would see whether these additional features could be accurately predicted. This would essentially work as a kind of model stacking, where we build a model to predict the training-set-only feature, then we use this predicted feature as a feature for predicting coupling constants.

There are eight different types of coupling constants in the dataset. Coupling constants are denoted xJyz, where x is the number of bonds that separate the two atoms of interest, y is the atom type for the first atom of interest (hydrogen, carbon or nitrogen) and z is the atom type for the second atom of interest. So 1JHN, for example, means the coupling constant is between a hydrogen and nitrogen bonded to each other, while 2JHC is for a hydrogen and carbon separated by one other atom (two bonds between them). The chart below illustrates the range of values observed in the training set for each type of coupling constant (outliers are diamonds).

![figure 1](fig1.png)

Clearly each type is a somewhat different critter, and that extends to which features are important. For example, the chart below illustrates the correlation between partial charge (calculated using the EEM method, there are different methods) and coupling constant for 1JHN, 2JHN and 3JHN.

![figure 2]( fig2.png)

There's lots of other examples to illustrate, but I think you can already see that relationships between features and coupling constants are highly type-specific. The biggest differences are between 1-bond, 2-bond and 3-bond. We could leave it to our model to figure this out, or we can train separate models for the different coupling constant types.

Clearly there are also slightly different things we need to consider for each flavor of coupling constant. For 1 bond, there are no atoms between the partners. For 2 bond, there is one atom between the partners and for 3 bond there are two atoms between the partners. The properties of the atoms between the partners are important. So 1-bond, 2-bond and 3-bond coupling constants are each in some ways a slightly different problem.

To deal with type-specific issues, I trained 8 networks, one for each flavor of coupling constant, and generated slightly different features for 1-bond, 2-bond and 3-bond couplings. This isn't as much work as it sounds, because we use the same network architecture for all 8 types, we just generate different features for each. Moreover, the competition assesses your score separately for each of the 8 coupling constants then averages them. If your performance is bad on one specific flavor of coupling constant, it's a lot easier to go back and retrain that one specific model than having to retrain a supersized model on 4.7 million datapoints. Finally, it's a lot faster to train if you can load the training set into memory. All in all, breaking things down into separate models looked like a good way to go.

Finally, data cleanup. I used the open-source package OpenBabel to convert the .xyz files containing the molecular structures to a file format that rdkit in Python can read. I checked the structures with rdkit and found some 200-odd unreadable (missing atoms, missing bonds, looks like a ball of yarn that got fed to a cat, etc.) Now, you could write a parser to go through and fix common types of errors, but then you have to get into the details of molecuar file format specifications, which is just more fun than you want to have on a Friday night. If there were a large number of problem children we'd have to do that. For a small handful of molecules, though, it's easier to just look at the structure of each problem child in a freeware chemistry drawing program (many available online), and if you see an obvious problem (usually it's pretty obvious) then fix it and save to a new sdf file.

Most of the problem structures were in the training set; we don't care about those because we have 4.7 million training datapoints and can afford to lose a tiny chunk. Test set, different story. I took the problem structures that were part of the test set and manually fixed those; there were only a few dozen of them and it was a quick fix.

## Strategy
A brief glance at the literature (Gilmer et al. 2017) suggests that for prediction of molecular properties, graph convolutional networks (GCNs) and related architectures win by wide margins. So as a starting approach, I built a GCN in PyTorch. It takes as input a feature vector for each atom and an adjacency matrix for the whole molecule, indicating which atoms are attached to which other atoms (we modify this adjacency matrix by adding 1 to diagonal elements so that the graph essentially includes a 'loop' edge connecting each atom to itself). We can normalize the adjacency matrix so that all rows / colums sum to 1, and I experimented with this; interestingly I achieved greater accuracy without adjacency matrix normalization. This is another aspect I would revisit if I had more time since this was unexpected and a little intriguing. Clearly we have some room for improvement on inter-layer normalization.

So for each datapoint, we have a feature matrix that is n x l, where l is the length of the feature vector and n is the number of atoms for the largest molecule in the dataset, and an n x n adjacency matrix. Both adjacency matrix and feature matrix are zero-padded to achieve size n. If we just feed the feature and adjacency matrices into the NN, though, the network doesn't know which atoms in this molecule we actually care about. The coupling constant is more strongly affected by atoms close to this pair of interest, so when generating the feature matrices I used one-hot encoding to indicate whether each atom was 1) one of the pair of interest, 2) attached to the pair of interest or in between them or 3) separated from the pair of interest by more than 1 bond (a "distant" atom). The architecture of the NN is diagrammed below (don't worry, I'll walk through how this works).

Now the model takes the input feature matrix and applies a linear layer followed by gated convolutional activation (see Dauphin et al., 2016 -- this outperforms ReLU here), giving us H1. Next, we multiply A H1 so that each atom is summed with its neighbors -- the atoms attached to it. We repeat this two-step process twice, so that each atom is being modified based on its neighbors which have now been modified based on THEIR neighbors. That's what graph convolution is all about. Finally, we use matrix multiplication to break the output of the last layer into categories 1 - 3 and sum all the atoms in each category, then normalize. At this point, a simple 3-layer fully-connected neural network with layer normalization converts this information into a predicted value. Calculate mean-squared-error loss, use the Adam gradient-descent based algorithm for optimization and there you go.

So what do we use for features? Given time restrictions I relied pretty heavily on domain knowledge to figure out which features might be useful. RDKit was a lifesaver here; nice documentation, easy to use, reads .sdf files and generates useful stuff (partial charges, distance matrix, what are the neighbors of a given atom etc.). I generate the features offline because generating them online would result in a glacially slow training / prediction process, then save them to numpy files we can memory-map if needed. After reading through relevant sections of the RDKit documentation and picking out features I thought would be helpful, I did a few experiments on 1JHH, 2JHN and 3JHH to narrow things down and get a feature set that worked well.

To pick hyperparameters I split each coupling constant training set up into training and validation sets using an 80-20 split (5x CV is not practical here). Layer normalization was hugely helpful, but not in the initial graph convolution layers. We ultimately got fairly negligible overfitting without using dropout or other regularization, suggesting that (again, given more time) we could have experimented with increased model capacity. Training for more than 40-50 epochs generally was not helpful, usually convergence was already achieved by that point. It is extremely important to note the dataset needs to be shuffled before training, otherwise each minibatch consists primarily of atom pairs from just two or three molecules, and the time needed for the network to converge increases dramatically. I trained this on GPU where it was pretty fast (I don't want to know how long it would have taken on CPU, not going to find out).

Finally, I tied this all together with a shell script. (Yes, I'm familiar with SnakeMake and NextFlow. No, I didn't think it was worthwhile to deploy them here.) Run the shell script without any arguments to see what it wants. It generates a feature set for the coupling type you specify, trains a neural network using the hyperparameters I selected and pickles the NN. The NN will print out what its loss looks like  Finally, if you ask nicely, it generates predictions using said NN for the test set. If you want to retrain the model for a specific coupling constant, piece of cake -- make the tweaks you want to make, run the shell script, go make yourself a cup of tea / grab a beer / talk to your partner / hang out with your dog, come back in 20. Good times.

Only thing to be aware of -- depending on how much RAM you have, the 2JHC and 3JHC datasets MAY be too big to fit in memory. Those are the two largest and the only ones where this is a possible concern. The easy fix is to memory-map the numpy feature files and rejigger the network to pull a minibatch at a time. I had 32GB of RAM (sweet luxury) and didn't have to do that, but YMMV. The smallest datasets are 1JHN, 2JHN and 3JHN, so if you want to play around with this those are a good place to start.

## Results
My results are tabulated below. Note that my model doesn't do too badly at all. In all categories, I achieve an R^2 > 0.999 and MAE as percent of average value for that category less than 0.001%. Depending on the nature of your goals, this might be fine. For a Kaggle competition, however, "good" is not good enough. We're in this to win, dammit. And my submission here was only in the top 40%, which is all right but nowhere near enough to grab some cash. But at this point I was out of time...


So what would I do differently? Clearly normalization in the network needed some work, we could possibly make use of the training-set-only features, and we could increase model capacity. I think the most room for improvement, however, lies in bond information. A brief glance at the literature (e.g. Duvenaud et al 2015) suggests including edge information is helpful and that's something we omitted here. My model uses the vertices of the graph, but makes no use of information about the edges, except insofar as they indicate who is attached to whom. Even more importantly, we make no use of the cartesian-coordinate distances between atoms, which I can 100% guarantee is important. I suspect that including that additional information would provide a bigger boost than further reconfiguring the architecture of the NN (although that is important too). Finally, I'd like to break performance down a little bit further -- are we generally making accurate predictions except where certain types of atom arrangements are concerned?

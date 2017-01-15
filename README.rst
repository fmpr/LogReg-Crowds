==============================================
LogReg-Crowds -- Logistic Regression from Crowds
==============================================

`LogReg-Crowds` is a collection of Julia implementations of various approaches for learning a logistic regression model multiple annotators and crowds, namely the works of:

* `Rodrigues, F., Pereira, F., and Ribeiro, B. Learning from multiple annotators: distinguishing good from random labelers. Pattern Recognition Letters, pp. 1428–1436, 2013 <http://www.fprodrigues.com/publications/learning-from-multiple-annotators-distinguishing-good-from-random-labelers/>`_.
* `Raykar, V., Yu, S., Zhao, L., Valadez, G., Florin, C., Bogoni, L., and Moy, L. Learning from Crowds. Journal of Machine Learning Research, pp. 1297–1322, 2010 <http://jmlr.csail.mit.edu/papers/volume11/raykar10a/raykar10a.pdf>`_.
* `Dawid, A. P. and Skene, A. M. Maximum likelihood estimation of observer error-rates using the EM algorithm. Journal of the Royal Statistical Society. Series C, 28(1):20–28, 1979 <https://www.jstor.org/stable/2346806>`_.

All implementations are able to handle multi-class problems and do not require repeated labelling (i.e. annotators do not have to provide labels for the entire dataset). The code was though for interpretability and it is well commented, so that it can be very easy to use (kindly see the file "demo.jl"). At the same, the Julia language provides it with a great perfomance, specially when compared to other scientific languages such as MATLAB or Python/Numpy, without compromising its high-level and interpretability.

Copyright (c) 2015 Filipe Rodrigues

This program is free software. You can redistribute it and/or modify it under the terms of the GNU General Public License, version 3, as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Appropriate reference to this software should be made when describing research in which it played a substantive role, so that it may be replicated and verified by others.

Basic Usage
------------

Include the LogReg-Crowds package::

    include("logreg_crowds.jl")
    using LogisticRegressionCrowds

Load the multi-annotator dataset::

    # an example multi-annotator dataset
    X = readdlm("data/fisheriris.csv", ',')
    Y = readdlm("data/fisheriris_labels_ma_missing.csv", ',')

Learn the model::

    est_weights, est_annotators_acc, est_groundtruth = LogisticRegressionCrowds.learn(X, Y, method="raykar", w_prior=1.0, pi_prior=0.01, max_em_iters=10)

Make predictions::

    predictions, predictive_probabilities = predict(X, est_weights)
    println("predictive accuracy: ", accuracy(predictions, y))

See the file "demo.jl" for a complete example.

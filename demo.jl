
include("logistic_regression.jl")
include("logreg_crowds.jl")

using LogisticRegression
using LogisticRegressionCrowds

srand(1) # fix seed

start_time = time() # to measure runtime

# read file with input vectors (NxD matrix)
X = readdlm("data/fisheriris.csv", ',')

# remove columns with zero standard deviation
stdX = std(X,1)
offset = 0
for i=1:size(stdX,2)
	if stdX[i] == 0
		X = [X[:,1:i-1+offset] X[:,i+1+offset:end]]
		offset -= 1
	end
end

# standardize inputs
#X = (X.-mean(X,1)) ./ std(X,1) 

# add bias term
#X = [ones(size(data,1)) X] 

# read file with ground truth labels (Nx1 vector) (optional)
y = readdlm("data/fisheriris_labels.csv", ',')[:]

# read file with annotators labels (NxR matrix; use -1 for missing answers)
#Y = readdlm("data/fisheriris_labels_ma.csv", ',')
Y = readdlm("data/fisheriris_labels_ma_missing.csv", ',') # version with missing labels

# read file with annotators' confusion matrices ((R*C)xC matrix) (optional)
confmat = readdlm("data/fisheriris_confmat.csv", ',')
R = int(confmat[1,1]); C = int(confmat[1,2])
confmat = reshape(float(confmat[2:end,:]'),C,C,R)
for r=1:R; confmat[:,:,r] = confmat[:,:,r]'; end

# run methods (uncomment the one you want to try out...)
#w = LogisticRegression.learn(X, y, w_prior=1.0)
w, est_annotators_acc, est_groundtruth, est_groundtruth_probs = LogisticRegressionCrowds.learn(X, Y, method="raykar", w_prior=1.0, pi_prior=0.01, groundtruth=y, max_em_iters=10)
#w, est_annotators_acc, est_groundtruth, est_groundtruth_probs = LogisticRegressionCrowds.learn(X, Y, method="rodrigues", w_prior=1.0, pi_prior=0.01, groundtruth=y, max_em_iters=10)
#w, est_annotators_acc, est_groundtruth, est_groundtruth_probs = LogisticRegressionCrowds.learn(X, Y, method="dawidskene", w_prior=1.0, pi_prior=0.01, groundtruth=y, max_em_iters=10)
#w, est_annotators_acc, est_groundtruth, est_groundtruth_probs = LogisticRegressionCrowds.learn(X, Y, method="majvote", w_prior=1.0, pi_prior=0.01, groundtruth=y)
#w, est_annotators_acc, est_groundtruth, est_groundtruth_probs = LogisticRegressionCrowds.learn(X, Y, method="naive", w_prior=1.0, pi_prior=0.01, groundtruth=y)

# make predictions
println("\napplying learned model")
predictions, predictive_probabilities = predict(X,w)
println("predictive accuracy: ", accuracy(predictions,y))

println("elapsed time: ", time()-start_time)


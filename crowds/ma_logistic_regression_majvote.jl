
module LogRegMajVote

include("../common.jl")

using Optim

function learn(_X, _Y; w_prior=1.0, pi_prior=0.01, groundtruth=nothing)

	# initialization
	println("\ninitializing...")
	global X = _X
	global Y = _Y
	global N = size(X,1)
	global D = size(X,2)
	global R = size(Y,2)
	classes = unique(Y[:])
	ix = findin(classes,-1)
	if size(ix,1) > 0
		classes = [classes[1:ix[1]-1]; classes[ix[1]+1:end]]
	end
	global C = size(classes,1)
	global M = C-1
	global V0 = w_prior
	global tau = pi_prior
	println("N=",N,"\nD=",D,"\nC=",C,"\nR=",R)

	global mN = 0.0000000000001*rand(D,M)

	# compute true annotators accuracies
	true_annotators_acc = nothing
	if groundtruth != nothing
		true_annotators_acc = annotators_accuracies(Y,groundtruth)
	end

	# estimate ground truth using majority voting
	println("\nestimating ground truth using majority voting")
	est_groundtruth, est_groundtruth_probs = majority_voting(Y)
	global latent_posterior = est_groundtruth_probs

	# evaluate estimated ground truth
	if groundtruth != nothing
		acc = accuracy(groundtruth,est_groundtruth)
		println("latent ground truth accuracy: ", acc)
	end

	# estimate annotators' reliabilities
	est_annotators_acc = annotators_accuracies(Y,est_groundtruth)
	
	# evaluate estimated annotators accuracies
	annotators_rmse = rmse(true_annotators_acc,est_annotators_acc)
	println("estimated annotators' accuracies rmse: ", annotators_rmse)

	# find MAP solution for weights for the function to return
	println("\nlearning logistic regression model over estimated ground truth")
	println("running l-BFGS...")
	res = optimize(negative_likelihood, gradient!, reshape(mN,D*M), method = :l_bfgs)
	println("f_minimum: ", res.f_minimum)
	mN = reshape(res.minimum,D,M)

	if groundtruth != nothing
		# evaluate estimated ground truth
		preds, probs = predict(X,mN)
		acc = accuracy(groundtruth,preds)
		println("logistic regression accuracy: ", acc)

		# evaluate estimated annotators accuracies
		est_annotators_acc = annotators_accuracies(Y,preds)
		annotators_rmse = rmse(true_annotators_acc,est_annotators_acc)
		println("estimated annotators' accuracies rmse: ", annotators_rmse)
	end

	return mN, est_annotators_acc, est_groundtruth, est_groundtruth_probs
end

function negative_likelihood(x::Vector)
	global N,D,M,X,V0,latent_posterior

	# compute posterior probabilities
	y_ik = ones(N,M+1)
	y_ik[:,1:M] = exp(X*reshape(x,D,M))
	y_ik = y_ik ./ repmat(sum(y_ik,2),1,M+1)
	
	# compute loglikelihood
	loglikelihood = sum(latent_posterior .* log(y_ik))

	# compute regularization term
	l2_regularization = sum(x.^2 / (2*(V0.^2)))
	loglikelihood -= l2_regularization
	
	#println("loglikelihood: ", loglikelihood)
	return -loglikelihood
end

function gradient!(x::Vector, storage::Vector)
	global D,M,X,V0,latent_posterior
	
	# compute posterior probabilities
	y_ik = exp(X*reshape(x,D,M))
	y_ik = y_ik ./ repmat(1.0+sum(y_ik,2),1,M)

	# compute gradients of loglikelihood
	storage[:] = -reshape(X' * (latent_posterior[:,1:M]-y_ik), D*M)

	# compute derivative of regularization term
	l2_regularization = x / (V0.^2)
	storage[:] += l2_regularization
end

end

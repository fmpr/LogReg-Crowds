
module LogRegNaive

include("../common.jl")

using Optim

function learn(X, Y; w_prior=1.0, pi_prior=0.01, groundtruth=nothing)

	# initialization
	println("\ninitializing...")
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

	# flatten Y
	global effective_N = 0
	for i=1:N
		for r=1:R
			if Y[i,r] != -1
				effective_N += 1
			end
		end
	end
	println("effective N=", effective_N)
	global X_flat = zeros(effective_N,D)
	global Y_flat = zeros(effective_N)
	global Y_flat_bin = zeros(effective_N,C)
	ind = 1
	for i=1:N
		for r=1:R
			if Y[i,r] != -1
				X_flat[ind,:] = X[i,:]
				Y_flat[ind] = Y[i,r]
				Y_flat_bin[ind,Y[i,r]] = 1
				ind += 1
			end
		end
	end

	# compute true annotators accuracies
	true_annotators_acc = nothing
	if groundtruth != nothing
		true_annotators_acc = annotators_accuracies(Y,groundtruth)
	end

	# find MAP solution for weights
	println("\nlearning logistic regression model using all labels from all annotators as training data")
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

	est_groundtruth, est_groundtruth_probs = predict(X,mN)
	est_annotators_acc = annotators_accuracies(Y,est_groundtruth)
	return mN, est_annotators_acc, est_groundtruth, est_groundtruth_probs
end

function negative_likelihood(x::Vector)
	global N,D,R,M,X_flat,Y_flat,V0,effective_N

	# compute posterior probabilities
	y_ik = ones(effective_N,M+1)
	y_ik[:,1:M] = exp(X_flat*reshape(x,D,M))
	y_ik = y_ik ./ repmat(sum(y_ik,2),1,M+1)
	
	# compute loglikelihood
	loglikelihood = sum(log(diag(y_ik[:,Y_flat])))

	# compute regularization term
	l2_regularization = sum(x.^2 / (2*(V0.^2)))
	loglikelihood -= l2_regularization
	
	#println("loglikelihood: ", loglikelihood)
	return -loglikelihood
end

function gradient!(x::Vector, storage::Vector)
	global D,M,X_flat,V0,Y_flat_bin
	
	# compute posterior probabilities
	y_ik = exp(X_flat*reshape(x,D,M))
	y_ik = y_ik ./ repmat(1.0+sum(y_ik,2),1,M)

	# compute gradients of loglikelihood
	storage[:] = -reshape(X_flat' * (Y_flat_bin[:,1:M]-y_ik), D*M)

	# compute derivative of regularization term
	l2_regularization = x / (V0.^2)
	storage[:] += l2_regularization
end

end

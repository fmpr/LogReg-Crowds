
module LogisticRegression

include("common.jl")

using Optim

function learn(_X, _y; w_prior=1.0)
	# initialization
	println("\ninitializing...")

	global X = _X
	global y = _y
	global N = size(X,1)
	global D = size(X,2)
	global C = size(unique(y),1)
	global M = C-1
	global V0 = w_prior
	println("N=",N,"\nD=",D,"\nC=",C)

	global mN = 0.0000000000001*rand(D,M)

	# build 1-of-K representation of the target variables y
	global y_bin = zeros(N,M)
	for i=1:N
		if y[i] < C
			y_bin[i,y[i]] = 1.0
		end
	end

	# find MAP solution
	println("\nlearning logistic regression model")
	res = optimize(negative_likelihood, gradient!, reshape(mN,D*M), method = :l_bfgs)
	println("f_minimum: ", res.f_minimum)
	mN = reshape(res.minimum,D,M)

	# make prediction and compute accuracy
	preds, probs = predict(X,mN)
	acc = accuracy(y,preds)
	class_accuracies(y,preds)
	println("global accuracy: ", acc)

	return mN
end

function negative_likelihood(x::Vector)
	global N,D,M,X,y,V0

	# compute posterior probabilities
	y_ik = ones(N,M+1)
	y_ik[:,1:M] = exp(X*reshape(x,D,M))
	y_ik = y_ik ./ repmat(sum(y_ik,2),1,M+1)
	
	# compute loglikelihood
	loglikelihood = sum(log(diag(y_ik[:,y])))

	# compute regularization term
	l2_regularization = sum(x.^2 / (2*(V0.^2)))
	loglikelihood -= l2_regularization
	
	#println("loglikelihood: ", loglikelihood)
	return -loglikelihood
end

function gradient!(x::Vector, storage::Vector)
	global D,M,X,V0,y_bin
	
	# compute posterior probabilities
	y_ik = exp(X*reshape(x,D,M))
	y_ik = y_ik ./ repmat(1.0+sum(y_ik,2),1,M)

	# compute gradients of loglikelihood
	storage[:] = -reshape(X' * (y_bin-y_ik), D*M)

	# compute derivative of regularization term
	l2_regularization = x / (V0.^2)
	storage[:] += l2_regularization
end

end

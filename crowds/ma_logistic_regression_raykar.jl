
module LogRegRaykar

include("../common.jl")

using Optim

function learn(_X, _Y; w_prior=1.0, pi_prior=0.01, groundtruth=nothing, max_em_iters=10)

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
	global latent_posterior = (1/C) * ones(N,C)
	global pi_est = zeros(C,C,R)
	for r=1:R
		pi_est[:,:,r] = eye(C,C) + tau # initialize with (near) identity matrix (i.e. assume annotator is reliable)
		pi_est[:,:,r] = pi_est[:,:,r] ./ repmat(sum(pi_est[:,:,r],2),1,C)
	end

	# compute true annotators accuracies
	true_annotators_acc = nothing
	if groundtruth != nothing
		true_annotators_acc = annotators_accuracies(Y,groundtruth)
	end

	# make a first evaluation before running EM
	if groundtruth != nothing
		println("\n*** iteration 0 ***")
		# evaluate estimated ground truth
		preds, probs = predict(X,mN)
		acc = accuracy(groundtruth,preds)
		println("logistic regression accuracy: ", acc)

		# evaluate estimated annotators accuracies
		est_annotators_acc = annotators_accuracies(Y,preds)
		annotators_rmse = rmse(true_annotators_acc,est_annotators_acc)
		println("estimated annotators' accuracies rmse: ", annotators_rmse)
	end

	# run EM
	for em_iter=1:max_em_iters
		println("\n*** iteration ", em_iter, " ***")

		# ------------------------- E-step
		println("E-step")

		complete_data_llikelihood = expected_loglikelihood(N,C,R,Y,latent_posterior,pi_est,mN)
		println("complete-data loglikelihood: ", complete_data_llikelihood)

		# compute posterior over latent variables (the ground truth labels)
		latent_posterior, preds = compute_posterior_over_y(N,C,R,Y,pi_est,mN)
		if groundtruth != nothing
			acc = accuracy(groundtruth,preds)
			println("latent ground truth accuracy: ", acc)
		end

		# ------------------------- M-step
		println("M-step")

		# find MAP solution for weights
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

		# estimate annotators' reliabilities
		pi_est = estimate_pi(N,C,R,Y,latent_posterior,tau)
	end

	est_groundtruth_probs, est_groundtruth = compute_posterior_over_y(N,C,R,Y,pi_est,mN)
	est_annotators_acc = annotators_accuracies(Y,est_groundtruth)
	return mN, est_annotators_acc, est_groundtruth, est_groundtruth_probs
end

function compute_posterior_over_y(N,C,R,Y,pi_est,mN)
	# compute logistic regression model posterior probabilities
	y_ik = ones(N,M+1)
	y_ik[:,1:M] = exp(X*mN)
	y_ik = y_ik ./ repmat(sum(y_ik,2),1,M+1)

	preds = zeros(N)
	for i=1:N
		adjustment_factor = ones(C,1)
		for r=1:R
			if Y[i,r] != -1
				adjustment_factor .*= pi_est[:,Y[i,r],r]
			end
		end
		y_ik[i,:] = adjustment_factor' .* y_ik[i,:]
		y_ik[i,:] = y_ik[i,:] ./ (sum(y_ik[i,:])*ones(1,C))

		preds[i] = indmax(y_ik[i,:])
	end

	return y_ik, preds
end

function estimate_pi(N,C,R,Y,latent_posterior,tau)
	pi_est = tau * ones(C,C,R)
	for r=1:R
		normalizer = zeros(C,1)
		for i=1:N
			if Y[i,r] != -1
				pi_est[:,Y[i,r],r] += latent_posterior[i,:]'
				normalizer += latent_posterior[i,:]'
			end
		end
		pi_est[:,:,r] = pi_est[:,:,r] ./ repmat(normalizer,1,C)
	end
	return pi_est
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

function expected_loglikelihood(N,C,R,Y,latent_posterior, pi_est, mN)
	# compute posterior probabilities
	y_ik = ones(N,M+1)
	y_ik[:,1:M] = exp(X*mN)
	y_ik = y_ik ./ repmat(sum(y_ik,2),1,M+1)

	complete_data_llikelihood = sum(latent_posterior .* log(y_ik))
	for i=1:N
		for r=1:R
			if Y[i,r] != -1
				complete_data_llikelihood += sum(latent_posterior[i,:] * log(pi_est[:,Y[i,r],r]))
			end
		end
	end

	return complete_data_llikelihood
end

end


module LogRegRodrigues

include("../common.jl")

using Optim

function learn(X, Y; w_prior=1.0, pi_prior=0.01, groundtruth=nothing, max_em_iters=10)

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

	global mN = 0.0000000000001*rand(D,M)
	global latent_posterior = ones(effective_N)
	global pi = 0.99 * ones(R)

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

		complete_data_llikelihood = expected_loglikelihood(X,Y,N,C,R,latent_posterior,pi,mN)
		println("complete-data loglikelihood: ", complete_data_llikelihood)

		# compute posterior over latent variables (the ground truth labels)
		latent_posterior = compute_posterior_over_z(X,Y,N,effective_N,C,R,pi,mN)

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
		pi = estimate_pi(X,Y,N,C,R,preds,tau)
	end

	est_groundtruth, est_groundtruth_probs = predict(X,mN)
	est_annotators_acc = annotators_accuracies(Y,est_groundtruth)
	return mN, est_annotators_acc, est_groundtruth, est_groundtruth_probs
end

function compute_posterior_over_z(X,Y,N,effective_N,C,R,pi,mN)
	# compute logistic regression model posterior probabilities
	y_ik = ones(N,M+1)
	y_ik[:,1:M] = exp(X*mN)
	y_ik = y_ik ./ repmat(sum(y_ik,2),1,M+1)

	gamma_z = zeros(effective_N)
	ind = 1
	for i=1:N
		for r=1:R
			if Y[i,r] != -1
				p_logreg = y_ik[i,Y[i,r]]
				p_rand = 1.0 / C
				gamma_z[ind] = p_logreg / (p_logreg + p_rand) # standard posterior over z
				#gamma_z[ind] = (pi[r] * p_logreg) / (pi[r] * p_logreg + (1-pi[r]) * p_rand) # prior-inflated posterior (using pi as a prior)
				ind += 1
			end
		end
	end

	return gamma_z
end

function estimate_pi(X,Y,N,C,R,preds,tau)
	# compute logistic regression model posterior probabilities
	#y_ik = ones(N,M+1)
	#y_ik[:,1:M] = exp(X*mN)
	#y_ik = y_ik ./ repmat(sum(y_ik,2),1,M+1)

	pi = tau * ones(R)
	for r=1:R
		normalizer = 0.0
		for i=1:N
			if Y[i,r] != -1
				# soft pi estimate
				#best = maximum(y_ik[i,:])
				#pi[r] += y_ik[i,Y[i,r]] * best
				#normalizer += best
				
				# hard pi estiamte
				if Y[i,r] == preds[i]
					pi[r] += 1
				end
				normalizer += 1
			end
		end
		pi[r] = pi[r] / normalizer
	end
	return pi
end

function negative_likelihood(x::Vector)
	global N,R,D,M,X_flat,Y_flat,V0,latent_posterior,effective_N

	# compute posterior probabilities
	y_ik = ones(effective_N,M+1)
	y_ik[:,1:M] = exp(X_flat*reshape(x,D,M))
	y_ik = y_ik ./ repmat(sum(y_ik,2),1,M+1)
	
	# compute loglikelihood
	loglikelihood = sum(latent_posterior .* log(diag(y_ik[:,Y_flat])))

	# compute regularization term
	l2_regularization = sum(x.^2 / (2*(V0.^2)))
	loglikelihood -= l2_regularization
	
	#println("loglikelihood: ", loglikelihood)
	return -loglikelihood
end

function gradient!(x::Vector, storage::Vector)
	global D,M,X_flat,Y_flat_bin,V0,latent_posterior
	
	# compute posterior probabilities
	y_ik = exp(X_flat*reshape(x,D,M))
	y_ik = y_ik ./ repmat(1.0+sum(y_ik,2),1,M)

	# compute gradients of loglikelihood
	storage[:] = -reshape(X_flat' * (repmat(latent_posterior,1,M) .* (Y_flat_bin[:,1:M]-y_ik)), D*M)

	# compute derivative of regularization term
	l2_regularization = x / (V0.^2)
	storage[:] += l2_regularization
end

function expected_loglikelihood(X,Y,N,C,R,latent_posterior, pi, mN)
	# compute posterior probabilities
	y_ik = ones(N,M+1)
	y_ik[:,1:M] = exp(X*mN)
	y_ik = y_ik ./ repmat(sum(y_ik,2),1,M+1)

	complete_data_llikelihood = 0
	ind = 1
	for i=1:N
		for r=1:R
			if Y[i,r] != -1
				complete_data_llikelihood += latent_posterior[ind]*log(y_ik[i,Y[i,r]])
				complete_data_llikelihood += (1-latent_posterior[ind])*log(1/C)
				ind += 1
			end
		end
	end

	return complete_data_llikelihood
end

end

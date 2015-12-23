
include("crowds/ma_logistic_regression_raykar.jl")
include("crowds/ma_logistic_regression_rodrigues.jl")
include("crowds/ma_logistic_regression_dawidskene.jl")
include("crowds/ma_logistic_regression_majvote.jl")
include("crowds/ma_logistic_regression_naive.jl")
include("common.jl")

module LogisticRegressionCrowds

using LogRegRaykar
using LogRegRodrigues
using LogRegDawidSkene
using LogRegMajVote
using LogRegNaive

function learn(X, Y; method="raykar", w_prior=1.0, pi_prior=0.01, groundtruth=nothing, max_em_iters=10)
	print("\nLearning logistic regression model from crowds using ")
	if lowercase(method) == "raykar"
		println("the approach of Raykar et al. (2010)")
		println("Learning From Crowds, Journal of Machine Learning Research, 2010.")
		println("http://delivery.acm.org/10.1145/1860000/1859894/11-1297-raykar.pdf")
		return LogRegRaykar.learn(X, Y; w_prior=w_prior, pi_prior=pi_prior, groundtruth=groundtruth, max_em_iters=max_em_iters)
	elseif lowercase(method) == "rodrigues"
		println("the approach of Rodrigues et al. (2013)")
		println("Learning from Multiple Annotators: Distinguishing Good from Random Labelers, Pattern Recognition Letters, 2013.")
		println("https://www.cisuc.uc.pt/publication/showfile?fn=1388438494_ma-lr.pdf")
		return LogRegRodrigues.learn(X, Y; w_prior=w_prior, pi_prior=pi_prior, groundtruth=groundtruth, max_em_iters=max_em_iters)
	elseif lowercase(method) == "dawidskene"
		println("the approach of Dawid and Skene (1979)")
		println("Maximum likelihood estimation of observer error-rates using the EM algorithm, Applied Statistics, 1979.")
		println("http://www.cs.mcgill.ca/~jeromew/comp766/samples/Output_aggregation.pdf")
		return LogRegDawidSkene.learn(X, Y; w_prior=w_prior, pi_prior=pi_prior, groundtruth=groundtruth, max_em_iters=max_em_iters)
	elseif lowercase(method) == "majvote"
		println("naive majority voting")
		return LogRegMajVote.learn(X, Y; w_prior=w_prior, pi_prior=pi_prior, groundtruth=groundtruth)
	elseif lowercase(method) == "naive"
		println("a naive logistic regression model that uses all the labels from all annotators for training")
		return LogRegNaive.learn(X, Y; w_prior=w_prior, pi_prior=pi_prior, groundtruth=groundtruth)
	else
		println("UNKNOWN!")
		println("Please select a valid method.")
	end
end


end

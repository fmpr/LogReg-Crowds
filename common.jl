
function accuracy(y,preds)
	return sum(y .== preds) / size(y,1)
end

function class_accuracies(y,preds)
	classes = unique(y[:])
	ix = findin(classes,-1)
	if size(ix,1) > 0
		classes = [classes[1:ix[1]-1]; classes[ix[1]+1:end]]
	end
	C = size(classes,1)
	accuracies = zeros(C)
	counts = zeros(C)
	for c=1:C
		for i=1:size(y,1)
			counts[y[i]] += 1.0
			if preds[i] == y[i]
				accuracies[y[i]] += 1.0
			end
		end
	end
	accuracies = accuracies ./ counts
	println("per-class accuracies:")
	for c=1:C
		println("class ",c,": ",accuracies[c])
	end
	return accuracies
end

function predict(X,mN)
	N = size(X,1)
	M = size(mN,2)
	probs = ones(N,M+1)
	probs[:,1:M] = exp(X*mN)
	probs = probs ./ repmat(sum(probs,2),1,M+1)
	preds = zeros(N)
	for i=1:N
		preds[i] = indmax(probs[i,:])
	end
	return preds, probs
end

function annotators_accuracies(Y,y)
	annotators_acc = zeros(size(Y,2))
	for r=1:size(Y,2)
		normalizer = 0
		for i=1:size(Y,1)
			if Y[i,r] != -1
				if Y[i,r] == y[i]
					annotators_acc[r] += 1.0
				end
				normalizer += 1.0
			end
		end
		annotators_acc[r] /= normalizer
	end
	return annotators_acc
end

function rmse(a,b)
	return sqrt(mean((a-b).^2))
end

function majority_voting(Y; prior=0)
	classes = unique(Y[:])
	ix = findin(classes,-1)
	if size(ix,1) > 0
		classes = [classes[1:ix[1]-1]; classes[ix[1]+1:end]]
	end
	C = size(classes,1)
	probs = prior * ones(size(Y,1),C)
	majvote = zeros(size(Y,1))
	for i=1:size(Y,1)
		normalizer = 0
		for r=1:size(Y,2)
			if Y[i,r] != -1
				probs[i,Y[i,r]] += 1.0
				normalizer += 1
			end
		end
		majvote[i] = indmax(probs[i,:])
		probs[i,:] = probs[i,:] / normalizer
	end
	return majvote, probs
end

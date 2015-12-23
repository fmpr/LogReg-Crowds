import sys
import numpy as np 

ground_truth_file = sys.argv[1]
confmat_file = sys.argv[2]

f = open(confmat_file)
R,C = f.readline().split(",")
R = int(R)
C = int(C)
annotators_confusion_matrices = np.zeros((R,C,C))
for r in xrange(R):
	for c in xrange(C):
		split = f.readline().replace("\n","").split(",")
		for c2 in xrange(C):
			annotators_confusion_matrices[r,c,c2] = float(split[c2])
f.close()

N = 0
annotators_accuracies = np.zeros(R)
f = open(ground_truth_file)
fw = open(ground_truth_file.replace(".csv", "")+"_ma.csv", "w")
for line in f:
	N += 1
	if not N % 100:
		print "\nsimulating annotators for instance no.", N
	true_label = int(line)
	print "true_label:", true_label
	for r in xrange(R):
		print annotators_confusion_matrices[r,true_label-1,:]
		simulated_label = np.random.multinomial(1, annotators_confusion_matrices[r,true_label-1,:]).argmax() + 1
		print "simulated_label for annotator %d: %d" % (r+1, simulated_label)
		if r != R-1:
			fw.write("%d," % (simulated_label,))
		else:
			fw.write("%d" % (simulated_label,))
		print simulated_label, true_label
		if simulated_label == true_label:
			annotators_accuracies[r] += 1.0
	fw.write("\n")
f.close()
fw.close()

print "\nFINAL STATS: "
annotators_accuracies /= N
print "R: ", R
print "C: ", C
print "N: ", N
print "annotators_accuracies: ", annotators_accuracies

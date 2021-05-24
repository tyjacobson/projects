# Tyler Jacobson

import getopt,sys
import pandas as pd 
import numpy as np
import math
import random

# For wednesday:
#	create 25-100 trees
#	get the root node
#	average out of bag set count
#	10:30

# Tree Node Class
class Node:
	
	# Initialization Function
	def __init__(self, motif, phi, fore, back):
		self.motif = motif
		self.phi = phi
		self.fore = fore
		self.back = back
		self.left = None
		self.right = None

# Binary Tree Class
class BTree:

	# Initialize Tree
	def __init__(self):
		self.root = None

# Randomly select 1/3 of sequences for test set
def rdm_select(df):

	test_set = df.sample(frac = 0.33333333, replace = False)
	test_set_names = list(test_set.index.values)
	train_set = df.drop(test_set_names)

	#print(test_set)
	#print(train_set)

	return test_set, train_set

# Tree Classifier
def classifier(tar, tree, test_set):

	#print(tar)

	# get all motifs
	motifs = test_set.columns.values.tolist()

	# get class data column
	class_dat = test_set.iloc[:, -1].tolist()

	# list of sequences in the test set
	seq_list = list(test_set.index.values)

	# get root data
	mot_indx = motifs.index(tree.root.motif)
	#print("Root: ", tree.root.motif)
	#seq_data = test_set.loc[tar, :].tolist()
	#print(seq_indx)
	#print(indx)

	#print(seq_data)

	#print(class_dat)

	seq_indx = seq_list.index(tar)
	seq_data = test_set.loc[tar, :].tolist()
	actual = class_dat[seq_indx]

	#print(actual)

	predict = 0
	#actual = 0

	# branch at root
	if seq_data[mot_indx] == 1:
		#print(seq_data[mot_indx])
		mot_indx = motifs.index(tree.root.left.motif)
		#print("LC: ", tree.root.left.motif)
		#print("1")
		# branch at left child
		if seq_data[mot_indx] == 1:
			#print(seq_data[mot_indx])
			#print('1')
			#mot_indx = motifs.index(tree.root.left.left.motif)
			if tree.root.left.left.fore > tree.root.left.left.back:
				#print('1')
				predict = 1
			else:
				#print('-1')
				predict = -1
		else:
			#print(seq_data[mot_indx])
			#print('0')
			#mot_indx = motifs.index(tree.root.left.right.motif)
			if tree.root.left.right.fore > tree.root.left.right.back:
				#print('1')
				predict = 1
			else:
				#print('-1')
				predict = -1
	else:
		#print(seq_data[mot_indx])
		mot_indx = motifs.index(tree.root.right.motif)
		#print(seq_data[mot_indx])
		#print("RC: ", tree.root.right.motif)
		#print("0")
		# branch at right child
		if seq_data[mot_indx] == 1:
			#mot_indx = motifs.index(tree.root.right.left.motif)
			#print('1')
			if tree.root.right.left.fore > tree.root.right.left.back:
				#print('1')
				predict = 1
			else:
				#print('-1')
				predict = -1
		else:
			#mot_indx = motifs.index(tree.root.right.right.motif)
			#print('0')
			if tree.root.right.right.fore > tree.root.right.right.back:
				#print('1')
				predict = 1
			else:
				#print('-1')
				predict = -1

	#print(predict, actual)

	return predict, actual


# Split the parent into a child of 1s and a child of 0s
def get_children(motif,df):

	# Get the motif
	#col_data = df.loc[:, motif].tolist()

	# split the parent
	x = df.loc[df[motif] == 1]
	y = df.loc[df[motif] == 0]
	
	#print(len(x))
	#print(x)

	#print(x.loc[:, motif].tolist())
	#print(y.loc[:, motif].tolist())
	# remove the motif used in the parent in the children
	x = x.drop(columns=[motif])
	y = y.drop(columns=[motif])

	#print(len(x))
	#print(x)

	return x, y

# Print the tree
def print_tree(tree, depth):

	#print(btree.root.motif, btree.root.phi)
	print("Root: ", tree.root.motif, tree.root.phi, tree.root.fore, tree.root.back)
	print("Left Child: ", tree.root.left.motif, tree.root.left.phi, tree.root.left.fore, tree.root.left.back)
	print("Right Child: ", tree.root.right.motif, tree.root.right.phi, tree.root.right.fore, tree.root.right.back)
	print("Left Left Child: ", tree.root.left.left.fore, tree.root.left.left.back)
	print("Left Right Child: ", tree.root.left.right.fore, tree.root.left.right.back)
	print("Right Left Child: ", tree.root.right.left.fore, tree.root.right.left.back)
	print("Right Right Child: ", tree.root.right.right.fore, tree.root.right.right.back)


	return 1

# record the motifs in the tree
#def record_tree(forest, depth):

#	for n in forest:
#		if 

# Find the root
def get_root(df):

	# get all motifs
	motifs = df.columns.values.tolist()
	del motifs[-1]	# remove the class column from list of motifs

	phi_values = []

	for n in motifs:
		ones, zeroes = get_children(n, df)
		#for m in ones:
		#	ones_ones, ones_zeroes = get_val(m, ones)
		#	zeroes_ones, zeroes_zeroes = get_val(m, zeroes)
		try:
			outer_phi = 2 * (len(ones) / len(df)) * (len(zeroes) / len(df))
			opos, oneg = get_class(ones)
			zpos, zneg = get_class(zeroes)
			inner_phi_pos = (len(opos) / len(ones)) - (len(zpos) / len(zeroes))
			inner_phi_neg = (len(oneg) / len(ones)) - (len(zneg) / len(zeroes))
			phi = (outer_phi * abs(inner_phi_pos - inner_phi_neg))
			phi_values.append(phi)
			#print(outer_phi, inner_phi_pos, inner_phi_neg, phi)
		except:
			phi_values.append(-1)


	#print(len(phi_values))

	best_phi = 0
	cnt = 0
	high_cnt = 0
	for i in phi_values:
		#print(i)
		if i > best_phi:
			best_phi = i
			high_cnt = cnt
		cnt += 1
	
	#print(tmp)
	#print(high_cnt)
	#print(motifs[high_cnt])
		#tmp = motifs
		#tmp.remove(n) 
		#for i in tmp:
		#	print(i)
		#print(ones)
		#print(zeroes)
		#break
		#print(ones, zeroes)

	best_motif = motifs[high_cnt]

	return best_motif, best_phi

# Create Tree Structure
def create_tree(df, depth):

	btree = BTree()

	# root motif
	root_motif, root_phi = get_root(df)
	fore, back = get_class(df)
	btree.root = Node(root_motif, root_phi, len(fore), len(back))

	# calculate children nodes
	lchild, rchild = get_children(btree.root.motif, df)

	# calculate the left child best motif
	left_motif, left_phi = get_root(lchild)
	lfore, lback = get_class(lchild)
	btree.root.left = Node(left_motif, left_phi, len(lfore), len(lback))

	# calculate the right child best motif
	right_motif, right_phi = get_root(rchild)
	rfore, rback = get_class(rchild)
	btree.root.right = Node(right_motif, right_phi, len(rfore), len(rback))


	# calculate the left child's children
	ll, lr = get_children(btree.root.left.motif, lchild)

	# calculat left child's left child best motif
	ll_m, ll_p = get_root(ll)
	llfore, llback = get_class(ll)
	btree.root.left.left = Node(ll_m, ll_p, len(llfore), len(llback))

	# calculat left child's right child best motif
	lr_m, lr_p = get_root(lr)
	lrfore, lrback = get_class(lr)
	btree.root.left.right = Node(lr_m, lr_p, len(lrfore), len(lrback))

	
	# calculate the right child's children
	rl, rr = get_children(btree.root.right.motif, rchild)

	# calculat right child's left child best motif
	rl_m, rl_p = get_root(rl)
	rlfore, rlback = get_class(rl)
	btree.root.right.left = Node(rl_m, rl_p, len(rlfore), len(rlback))

	# calculat right child's right child best motif
	rr_m, rr_p = get_root(rr)
	rrfore, rrback = get_class(rr)
	btree.root.right.right = Node(rr_m, rr_p, len(rrfore), len(rrback))

	#print(btree.root.motif, btree.root.phi)

	return btree

# Calculate the number of positive/negative class sequences in a motif
def get_class(motif):

	pos = motif.loc[motif["Class"] == 1]
	neg = motif.loc[motif["Class"] == -1]

	return pos, neg

# bootstrap, returns bootstrapped data set and a random subset
# 	of sqrt(size of # of motifs) motifs
def bootstrap(df):

	col_names = list(df.columns.values)
	#del col_names[-1] #remove class column from list of motifs

	boot_set = df.sample(frac = 1, replace = True)
	boot_set_names = list(boot_set.index.values)
	
	del col_names[-1] #remove class column from list of motifs

	bag_set = df.drop(boot_set_names)
	#for n in boot_set_names:
	#	print(n)

	rdm_sub = []

	for m in range(int(math.sqrt(len(col_names)))):
		while(1):
			x = random.randrange(len(col_names))
			try:
				rdm_sub.index(col_names[x])
			except:
				rdm_sub.append(col_names[x])
				break

	#print("Random Subset:")
	#for n in rdm_sub:
	#	print(n)
	
	# Re-add the class column back into the rdm subset
	rdm_sub.append("Class")

	return boot_set, rdm_sub, bag_set

# Get the data set for the bootstrap data set and the selected motifs
def fit_df(boot_set, rdm_sub):

	fit_boot = boot_set[rdm_sub]
	#fit_names = list(fit_boot.columns.values)
	#print(rdm_sub)
	#print(fit_names)
	#print(fit_boot)

	return fit_boot

# print results
def print_result(eval_results):
	# Calculate the Data
	total_acc = []
	total_err = []
	total_pfp = []
	total_pfn = []
	total_sens = []
	total_spec = []
	total_tp = []
	total_fp = []
	total_tn = []
	total_fn = []
	for x in range(len(eval_result)):
		TP = eval_result[x][0]
		FP = eval_result[x][1]
		TN = eval_result[x][2]
		FN = eval_result[x][3]
		print("Iteration ", x + 1, ":")
		
		# TP
		total_tp.append(TP)
		# FP
		total_fp.append(FP)
		# TN
		total_tn.append(TN)
		# FN
		total_fn.append(FN)

		# TP/FP/TN/FN
		print("TP: ", TP, "FP: ", FP, "TN: ", TN, "FN: ", FN)

		# Accuracy
		accuracy = ((TN + TP)/(TN + FN + FP + TP))
		total_acc.append(accuracy)
		print("Accuracy: ", accuracy)

		# Error Rate
		err_rate = 1 - accuracy
		total_err.append(err_rate)
		print("Error Rate: ", err_rate)

		# PFP
		pfp = (FP / (FP + TP))
		total_pfp.append(pfp)
		print("PFP: ", pfp)

		# PFN
		pfn = (FN / (FN + TN))
		total_pfn.append(pfn)
		print("PFN: ", pfn)

		# Sensitivity
		sens = (TP / (TP + FN))
		total_sens.append(sens)
		print("Sensitivity: ", sens)

		# Specificity
		spec = (TN / (FP + TN))
		total_spec.append(spec)
		print("Specificity: ", spec)

		print()

	# Print Averages
	print("Averages:")
	print("TP: ", (total_tp[0] + total_tp[1] + total_tp[2]) / 3)
	print("FP: ", (total_fp[0] + total_fp[1] + total_fp[2]) / 3)
	print("TN: ", (total_tn[0] + total_tn[1] + total_tn[2]) / 3)
	print("FN: ", (total_fn[0] + total_fn[1] + total_fn[2]) / 3)

	print("Accuracy: ", (total_acc[0] + total_acc[1] + total_acc[2]) / 3)
	print("Error Rate: ", (total_err[0] + total_err[1] + total_err[2]) / 3)
	print("PFP: ", (total_pfp[0] + total_pfp[1] + total_pfp[2]) / 3)
	print("PFN: ", (total_pfn[0] + total_pfn[1] + total_pfn[2]) / 3)
	print("Sensitivity: ", (total_sens[0] + total_sens[1] + total_sens[2]) / 3)
	print("Specificity: ", (total_spec[0] + total_spec[1] + total_spec[2]) / 3)


if __name__ =='__main__':
	#df = pd.read_csv('ets.csv',index_col = 0)
	df = pd.read_csv('SREBP.training.csv',index_col = 0)
	pd.set_option('display.max_rows', df.shape[0]+1)
	#pd.set_option('display.max_columns', df.shape[1]+1)
	pd.options.display.max_colwidth = 200

	#boot_set, rdm_sub, bag_set = bootstrap(df)
	#fit_boot = fit_df(boot_set, rdm_sub)

	depth = 3

	eval_result = []
	# 3-fold cross-validation
	for x in range(3):
		# Forest Array
		forest= []
		forest_size = 25
		bag_count = []
		# get the test set and training set
		test_set, train_set = rdm_select(df)

		#Create the forest
		for n in range(forest_size):
			# create the tree using the fit_boot data set
			boot_set, rdm_sub, bag_set = bootstrap(train_set)
			fit_boot = fit_df(boot_set, rdm_sub)
			dtree = create_tree(fit_boot, depth)
			forest.append(dtree)
			bag_count.append(len(bag_set))
			#print_tree(dtree, depth)
			#record_tree(dtree, depth)

		#print_tree(dtree, depth)

		# get a list of the motifs used in the test set
		test_seq = list(test_set.index.values)

		# put the classifier here. Take majority of classifier votes to
		#   classify the selected sequence.
		#tar = "chr12:24720923-24721791"	#srebp
		#tar = "chr19:42364200-42364470" #ets
		TP = 0
		TN = 0
		FP = 0
		FN = 0
		for z in test_seq:
			pos_count = 0
			neg_count = 0
			for m in range(len(forest)):
				#print(forest[m].root.motif, "OoB Count: ", bag_count[m])
				try:
					predict, actual = classifier(z, forest[m], df)
					if predict == 1:
						pos_count += 1
					else:
						neg_count += 1
				except:
					print("Sequence not in Dataset")

			#print("Positive Count: ", pos_count)
			#print("Negative Count: ", neg_count)
			if pos_count > neg_count:
				#print("Classified: +1")
				if actual == 1:
					TP += 1
				else:
					FP += 1
			else:
				#print("Classified: -1")
				if actual == 1:
					FN += 1
				else:
					TN += 1

			oob_average = 0

			for i in bag_count:
				oob_average += i

			#print("OoB Average: ", oob_average/forest_size)

		val = []
		val.append(TP)
		val.append(FP)
		val.append(TN)
		val.append(FN)

		eval_result.append(val)

	print_result(eval_result)
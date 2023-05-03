Download Link: https://assignmentchef.com/product/solved-cs273a-homework-4
<br>
<h2>1 Setting up the data</h2>

The following is the snippet of code to load the datasets, and split it into train and validation data:

<table width="624">

 <tbody>

  <tr>

   <td width="624"># Data LoadingX    = np.genfromtxt(‘data/X_train.txt’, delimiter=None)Y    = np.genfromtxt(‘data/Y_train.txt’, delimiter=None)X,Y = ml.shuffleData(X,Y)</td>

  </tr>

 </tbody>

</table>

1

2

3

4

<ol>

 <li>Print the minimum, maximum, mean, and the variance of all of the features. <em>5 points</em></li>

 <li>Split the dataset, and rescale each into training and validation, as:</li>

</ol>

<table width="591">

 <tbody>

  <tr>

   <td width="591">Xtr, Xva, Ytr, Yva = ml.splitData(X, Y)Xt, Yt = Xtr[:5000], Ytr[:5000] # subsample for efficiency (you can go higher)XtS, params = ml.rescale(Xt)                                       # Normalize the featuresXvS, _ = ml.rescale(Xva, params) # Normalize the features</td>

  </tr>

 </tbody>

</table>

1

2

3

4

Print the min, maximum, mean, and the variance of the rescaled features. <em>5 points</em>

<h2>       2        Linear Classifiers</h2>

In this problem, you will use an existing implementation of logistic regression, from the last homework, to analyze its performance on the Kaggle dataset.

<table width="624">

 <tbody>

  <tr>

   <td width="624">learner = mltools.linearC.linearClassify() learner.train(XtS, Yt, reg=0.0, initStep=0.5, stopTol=1e-6, stopIter=100) learner.auc(XtS, Yt) # train AUC</td>

  </tr>

 </tbody>

</table>

1

2

3

<ol>

 <li>One of the important aspects of using linear classifiers is the regularization. Vary the amount of regularization, reg , in a wide enough range, and plot the training and validation AUC as the regularization weight is varied. Show the plot. <em>10 points</em></li>

 <li>We have also studied the use of polynomial features to make linear classifiers more complex. Add degree 2 polynomial features, print out the number of features, why it is what it is. <em>5 points</em></li>

 <li>Reuse your code that varied regularization to compute the training and validation performance (AUC) for this transformed data. Show the plot. <em>5 points</em></li>

</ol>

<h2>       3        Nearest Neighbors</h2>

In this problem, you will analyze an existing implementation of K-Nearest-neighbor classification for the Kaggle dataset. The K-nearest neighbor classifier implementation supports two hyperparameters: the size of the neighborhood, <em>K</em>, and how much to weigh the distance to the point, <em>a </em>(0 means no unweighted average, and the higher the <em>α</em>, the higher the closer ones are weighted<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>). Note, you might have to subsample a lot for KNN to be efficient.

<table width="624">

 <tbody>

  <tr>

   <td width="624">learner = mltools.knn.knnClassify() learner.train(XtS, Yt, K=1, alpha=0.0) learner.auc(XtS, Yt) # train AUC</td>

  </tr>

 </tbody>

</table>

1

2

3

<ol>

 <li>Plot of the training and validation performance for an appropriately wide range of <em>K</em>, with <em>α</em>= 0. <em>5 points</em></li>

 <li>Do the same with unscaled/original data, and show the plots. <em>5 points</em></li>

 <li>Since we need to select both the value of <em>K </em>and <em>α</em>, we need to vary both, and see how the performance changes. For a range of both <em>K </em>and <em>α</em>, compute the training and validation AUC (for unscaled or scaled data,</li>

</ol>

whichever you think would be a better choice), and plot them in a two dimensional plot like so:

<table width="591">

 <tbody>

  <tr>

   <td width="591">K = <strong>range</strong>(1,10,1) # Or something else A = <strong>range</strong>(0,5,1) # Or something else tr_auc = np.zeros((<strong>len</strong>(K),<strong>len</strong>(A))) va_auc = np.zeros((<strong>len</strong>(K),<strong>len</strong>(A))) <strong>for </strong>i,k <strong>in enumerate</strong>(K):<strong>for </strong>j,a <strong>in enumerate</strong>(A):tr_auc[i][j] = … # train learner using k and a va_auc[i][j] = … # Now plot it f, ax = plt.subplots(1, 1, figsize=(8, 5)) cax = ax.matshow(mat, interpolation=’nearest’) f.colorbar(cax) ax.set_xticklabels([”]+A) ax.set_yticklabels([”]+K) plt.show()</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

Show both the plots, and recommend a choice of <em>K </em>and <em>α </em>based on these results. <em>10 points</em>

<h2>       4        Decision Trees</h2>

For this problem, you will be using a similar analysis of hyper-parameters for the decision tree implementation.

<table width="57">

 <tbody>

  <tr>

   <td width="57">maxDepth</td>

  </tr>

 </tbody>

</table>

There are three hyper-parameters in this implementation that become relevant to its performance;, minParent , and minLeaf , where the latter two specify the minimum number of data points necessary to split a

node and form a node, respectively.

<table width="624">

 <tbody>

  <tr>

   <td width="624">learner = ml.dtree.treeClassify(Xt, Yt, maxDepth=15)</td>

  </tr>

 </tbody>

</table>

1

<table width="57">

 <tbody>

  <tr>

   <td width="57">maxDepth</td>

  </tr>

 </tbody>

</table>

<ol>

 <li>Keeping minParent=2 and minLeaf=1 , varyto a range of your choosing, and plot the training and validation AUC. <em>5 points</em></li>

 <li>Plot the number of nodes in the tree as maxDepth is varied (using sz ). Plot another line in this plot by increasing either minParent or minLeaf (choose either, and by how much). <em>5 points</em></li>

</ol>

<table width="57">

 <tbody>

  <tr>

   <td width="57">maxDepth</td>

  </tr>

 </tbody>

</table>

<ol start="3">

 <li>Setto a fixed value, and plot the training and validation performance of the other two hyperparameters in an appropriate range, using the same 2D plot we used for nearest-neighbors. Show the plots, and recommend a choice for minParent and minLeaf based on these results. <em>10 points</em></li>

</ol>

<h2>       5        Neural Networks</h2>

Last we will explore the use of neural networks for the same Kaggle dataset. The neural networks contain many possible hyper-parameters, such as the number of layers, the number of hidden nodes in each layer, the activation function the hidden units, etc. These don’t even take into account the different hyper-parameters of the optimization algorithm.

<table width="624">

 <tbody>

  <tr>

   <td width="624">nn = ml.nnet.nnetClassify() nn.init_weights([[XtS.shape[1], 5, 2], ‘random’, XtS, Yt) # as many layers nodes you want nn.train(XtS, Yt, stopTol=1e-8, stepsize=.25, stopIter=300)</td>

  </tr>

 </tbody>

</table>

1

2

3

<ol>

 <li>Vary the number of hidden layers and the nodes in each layer (we will assume each layer has the same number of nodes), and compute the training and validation performance. Show 2D plots, like for decision trees and K-NN classifiers, and recommend a network size based on the above.</li>

 <li>Implement a new activation function of your choosing, and introduce it as below:</li>

</ol>

<table width="591">

 <tbody>

  <tr>

   <td width="591"><strong>def </strong>sig(z): <strong>return </strong>np.atleast_2d(z) <strong>def </strong>dsig(z): <strong>return </strong>np.atleast_2d(1) nn.setActivation(‘custom’, sig, dsig)</td>

  </tr>

 </tbody>

</table>

1

2

3

<table width="142">

 <tbody>

  <tr>

   <td width="57">logistic</td>

   <td width="29">and</td>

   <td width="57">htangent</td>

  </tr>

 </tbody>

</table>

Compare the performance of this activation function with, in terms of the training and validation performance.

<h2>6       Conclusions</h2>

Pick the classifier that you think will perform best, mention all of its hyper-parameter values, and explain the reason for your choice. Train it on as much data as you can, preferably all of X , submit the predictions on Xtest to Kaggle, and include your Kaggle username and leaderboard AUC in the report. Here’s the code to create the Kaggle submission:

<table width="624">

 <tbody>

  <tr>

   <td width="624">Xte = np.genfromtxt(‘data/X_test.txt’, delimiter=None) learner = .. # train one using X,YYte = np.vstack((np.arange(Xte.shape[0]), learner.predictSoft(Xte)[:,1])).T np.savetxt(‘Y_submit.txt’, Yte, ‘%d, %.2f’, header=’ID,Prob1′, comments=”, delimiter=’,’)</td>

  </tr>

 </tbody>

</table>

1

2

3

4

<h2>Statement of Collaboration</h2>

It is <strong>mandatory </strong>to include a <em>Statement of Collaboration </em>in each submission, with respect to the guidelines below. Include the names of everyone involved in the discussions (especially in-person ones), and what was discussed.

All students are required to follow the academic honesty guidelines posted on the course website. For programming assignments, in particular, I encourage the students to organize (perhaps using Campuswire) to discuss the task descriptions, requirements, bugs in my code, and the relevant technical content <em>before </em>they start working on it. However, you should not discuss the specific solutions, and, as a guiding principle, you are not allowed to take anything written or drawn away from these discussions (i.e. no photographs of the blackboard,

written notes, referring to Campuswire, etc.). Especially <em>after </em>you have started working on the assignment, try to restrict the discussion to Campuswire as much as possible, so that there is no doubt as to the extent of your collaboration.
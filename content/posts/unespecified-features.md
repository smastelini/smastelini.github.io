---
title: "Unespecified features and the road so far"
date: 2022-07-19
draft: false
tags: ["regression", "river", "hoeffding-trees"]
type: "post"
---
    
Maintaining a machine learning (ML) library sometimes makes you make tough decisions and take audacious paths. Those are the perks of creating something so close to research. Being a full-time ML researcher, I got used to the "rules of the game." We do a literature review, find a good research gap, create relevant research questions, and develop some hypotheses. Then it is a matter of structuring your methodological setup to evaluate the proposed solutions. Sometimes we delve into theoretical stuff. Sometimes it is more empirical. If everything goes well, we write a manuscript that gets accepted after some time. And that is pretty much it. Do it again and again.

However, practice and theory not always are on good terms. Writing and testing code is usually more dynamic than the publishing process I described before. In (online) ML, there are so many details in an algorithm that perhaps they cannot fit in the paper format. Maybe they even should not. I really don't have a definitive answer to that.
Sometimes things just evolve, just like online machine learning models, right?

Some time ago, I came across an [interesting paper](https://arxiv.org/abs/2010.08199) that talks about something a bit unusual. In this paper, Manapragada et al. talk about some unspecified changes (or tweaks, one may say) that have been incorporated into the code of incremental decision tree models. They evaluate how such changes impact the final performance of the models. I had always wondered about those design decisions I could not find in the original research papers on the tree algorithms (and other models). After working with incremental decision trees for more than four years now, I have reached similar conclusions to those exposed by the authors after spending hours and hours reading, debugging, and trying to understand what was going on in the existing code of libraries such as [MOA](https://moa.cms.waikato.ac.nz/) and [skmultiflow](https://scikit-multiflow.github.io/). And I mean, not everyone that uses decision trees or any other learning model keeps an ML library, so they are not supposed to know these kinds of details if they are not specified somewhere other than the code. So, I think what Maganaprada et al. did is relevant for researchers and practitioners.

During my time at (the now discontinued) skmultiflow, I always tried to add some commentary to "fuzzy code portions" every time I figured them out. Some key extra functionalities of models, such as Hoeffding Adaptive Trees, not described in the original paper, took me years to understand (more about that later). And I don't judge the developers/researchers by any means. Sometimes we need just a little more time to think about something even better than we had previously thought and published. And not always this "something better" is paper-worthy. That's why I believe Magapragada's paper is really relevant.


## 1. My part in the extra tweaks in all these years

I'm also to blame for things that are not written anywhere except in the code. In my defense, I always tried to document these things, and I believe they bring something valuable to the end user. Let me list some examples, and later I will talk about something more recent and perhaps more significant I have done. I always wanted to create a record of these things somewhere, but my thesis topic and my current time schedule don't allow me to fit all these tweaks in the formal structure of a research paper. A blog post might fit the bill.

Some models in [River](https://riverml.xyz) have perks you cannot find anywhere else, as far as I am concerned. These things make me really happy, even though many of them are simple modifications and Quality of Life (QoL) improvements.

Following, I list a somewhat long and tedious list of things I implemented in skmultiflow and River that are not found in research papers. I will not discuss them in depth, as my main objective here is to create a log of unspecified behaviors. If somebody is interested in expanding these points in the form of documentation or even a paper, please e-mail me. We can collaborate on that 🙂.


### 1.1. Memory management in Hoeffding Tree regressors

With the help of professor Bernhard Pfahringer and Jacob Montiel, I adapted the memory management routines proposed by Richard Brendon Kirkby in his [thesis](https://researchcommons.waikato.ac.nz/handle/10289/2568) to Hoeffing Tree (HT) regressors. In the original proposal, Kirkby worked with classification trees and used a distribution impurity heuristic to rank nodes and make the least promising ones stop attempting to split.

In River, regression tree nodes are ranked by their depth, and the deepest ones might get deactivated if the memory limit is reached. By deactivated, I mean they will not be allowed to split and monitor split-enabling statistics to save memory. You can check the PR I created in skmultiflow [here](https://github.com/scikit-multiflow/scikit-multiflow/pull/230).

### 1.2. Attribute splitter improvements

I have simplified the original regression trees' attribute splitter (Extended Binary Search Tree – E-BST) to avoid monitoring redundant target statistics and, thus, saving some memory. This idea is described in the [Ph.D. thesis](http://kt.ijs.si/theses/phd_aljaz_osojnik.pdf) of Aljaž Osojnik. You can check the PR [here](https://github.com/scikit-multiflow/scikit-multiflow/pull/237).

In this same PR, I also brought to life the memory management routine proposed in the [original paper](https://kt.ijs.si/elena_ikonomovska/DAMI10.pdf) about HT regressors, authored by Ikonomovska et al. The original proposal was based on the inefficient version of E-BST, so I had to modify the algorithm slightly to work with the optimized E-BST version. 

Although these modifications do not change how the trees are used, the models become faster and more lightweight "for free." The plus side: there are no impacts on the predictive performance.

As far as I am concerned, HT regressors in MOA do not have the same capabilities.

### 1.3. Adaptive Random Forest regressor tweaks

With the approval of Heitor Murilo Gomes, I added the option of using the median statistic to aggregate the predictions of individual trees in the Adaptive Random Forest (ARF) regressor (and Streaming Random Patches). In a [paper](https://ieeexplore.ieee.org/abstract/document/9206756) in collaboration with Heitor and other authors, we have found that combining median aggregation with model trees is often beneficial for predictive performance.

In the original MOA code, the raw output of the trees is monitored by the drift detectors, as far as I am concerned. Additionally, the ARF regressor in River monitors the absolute error of individual trees to track concept drifts. I believe that tracking errors make more sense. You can check it yourself and compare the ARF for regression in MOA against the River version. Time differences aside, in my tests, usually River has the edge concerning the predictive performance!

### 1.4. QoL improvements to Hoeffding Trees

Allowing models to grow indefinitely in streaming scenarios is not always a good idea. During the merger of creme and skmultiflow, I made sure all the HT models in River had the option to limit their maximum height. In the case of HT regressors, I also added the option of using adaptive leaf models in single-target regression, just like it was proposed for multi-target trees (in the latter case, adaptive leaf models are described in research papers).

After a long time of dreaming about it, I ensured that the user could use any regressor as the leaf model instead of being limited to linear regression as in skmultiflow and MOA. Just imagine! One can use a k-Nearest Neighbor or a time series model to output predictions in the leaves of a regression tree. Crazy huh? Unfortunately, I don't have enough time to experiment with this weird stuff, but I would love to see this potential applied in practice. If somebody does that, please let me know!

I have always seen HTs' components as lego blocks. You can pick different things and fit them together to build varied stuff. Since my time at skmultiflow and now River, I have done a lot of waves of refactoring to make the original code (ported before my time from MOA) more and more modular. I'm pretty happy with the results and believe the current models are "lego-esque" already. But I still have work to do on that. Someday I might post something further discussing this topic.

### 1.5. Option trees... of sorts

Just like it is implemented in MOA (also as an unspecified feature) and described in Manapragada's paper, all Hoeffding Adaptive Trees (HAT) can act like option trees, kind of. HAT creates alternate tree branches every time drift is detected in a subtree. In MOA, if the subtree is big enough, it can contribute to the final predictions, even though the old subtree was not discarded. In other words, MOA's HAT combines the outputs of the main subtree and the alternate one to compose the final class probabilities.

Nonetheless, this only happens if the alternate tree is not composed of a single node, i.e., it has made some splits. This is not detailed in the HAT paper. In River, I have decided to take a different path. River's HAT always combines the predictions of the main tree with the alternate one, even if the alternate tree is simply a leaf. I think the most recent subtree always has something to bring to the table, as it is induced using only the data arriving after the concept drift.

### 1.6. Missing and emerging data in Hoeffding Trees

One of the most beautiful things about using dictionaries as the input data format in River is being able to explore sparsity. You should try it!

HTs in River are robust to missing data and emerging features. After some discussion with Jacob Montiel and Max Halford, we have settled on using the following standards for all the decision trees:

1. **Missing data:** if a feature value is used to split and is missing in an instance, we choose the most common path in the tree to sort the data further. This applies to both learning and inference.
2. **Emerging features:** we update the split statistics of the leaves to account for the new feature.
3. **New categorical value:**
    - If this feature is used in a split, we add a new branch for the new value;
    - Otherwise, we update the split statistics of the leaves.
        

## 2. Hoeffding Adaptive Tree regressor, previously known as my nemesis

When I started collaborating in skmultiflow, I was pretty new to online ML and HTs. By then, I soon discovered something that has haunted me for a long time: the Hoeffding Adaptive Tree regressor.

Let's not rush things so you can understand my point. HAT is a wonder proposed for classification proposed by [Bifet and Gavalda](https://link.springer.com/chapter/10.1007/978-3-642-03915-7_22) quite some time ago. It shares the basic HT framework with plenty of trees, but some essential differences exist. These differences allow HAT to leverage the best from HTs and tackle their primary weakness: the inability to deal with non-stationary distributions.

And there are even some unspecified extra features I have mentioned before and are better discussed in the paper by Manapragada et al.

### 2.1. HAT in a nutshell:

Each non-leaf node in HAT carries a concept drift detector. If this detector signals a drift at any given time, HAT starts a new subtree rooted at this node. This alternate or background subtree then keeps learning from the new data (after the detected drift) until a user-given threshold is reached. The foreground tree also keeps learning.

After the warm-up period, defined by the threshold aforementioned, HAT compares the foreground subtree against the background one and only keeps the best option. If the background subtree is the best option, it replaces the foreground tree, and the outdated subtree is discarded. If the older subtree still has its mojo, i.e., it is still the most accurate one, the background tree is discarded.

### 2.2. The problem and the elephant in the room

Here is the thing: the meaning of "best" is defined via a statistical test. This literally took me years to figure out. I cannot count how many times I have read the following portion of the HAT classifier code without not getting its meaning ([permalink](https://github.com/online-ml/river/blob/6f390b51b25f87989b463e11bd0460a9bf83e069/river/tree/nodes/hatc_nodes.py#L215-L229)):


```python
            if alt_n_obs > tree.drift_window_threshold and n_obs > tree.drift_window_threshold:
                old_error_rate = self._mean_error.get()
                alt_error_rate = self._alternate_tree._mean_error.get()


                n = 1.0 / alt_n_obs + 1.0 / n_obs


                bound = math.sqrt(
                    2.0
                    * old_error_rate
                    * (1.0 - old_error_rate)
                    * math.log(2.0 / tree.switch_significance)
                    * n
                )


                if bound < (old_error_rate - alt_error_rate):
                    # it goes on from here
```

It took me eons (and a confirmation e-mail from professor Albert Bifet, which kindly answered me confirming my hypothesis) to understand that the above expressions use a statistical bound derived from the Hoeffding bound and one property of Bernoulli distributions.

Wait, what?

Well, let me give you a little more context. Most supervised and non-stationary online ML models monitor some type of error to detect concept drifts. In the classification case, the usual strategy is to use a \\(0-1\\) loss. In other words, the drift detector receives \\(0\\) if the node correctly classifies an instance and \\(1\\) otherwise. Therefore, we monitor a random Bernoulli variable. Two variables, if we think about it. One for the foreground tree and another one for the background tree. This \\(0-1\\) loss is precisely what we compare to decide which is the best subtree to keep. Hence, the idea of using a frequentist confidence interval test based on the Hoeffding inequality. I finally got this part after reading [this](https://en.wikipedia.org/w/index.php?title=Hoeffding%27s_inequality&section=6#Confidence_intervals) and spending a lot of time thinking. I just needed four years to understand this idea :sweat_smile:.

The HAT authors do not explicitly mention this specific type of test in the paper, but they note that any proper test could be used with HAT. The Hoeffing inequality-based test (using the Bernoulli variable properties) is what made its way into the HAT code. So far, so good. This is what HAT was initially designed for. But there was a problem that haunted me all those years. MOA does not have it; skmultiflow had: Hoeffding Adaptive Tree regressor. I sincerely do not know its origin nor who initially made a tentative implementation of it. The fact is that HATR, let's call it like that, was already in skmultiflow when I arrived and it makes so much sense to exist!

HAT for classification is very nice, so why not use it in regression? The problem is that it is not formally proposed anywhere. By tentative implementation, I mean that the original code did not work. When I tried it, there were a lot of numerical exceptions going on and bugs.

Other collaborators and I have fixed the bugs throughout the years. But the specific testing I presented above for HAT was still to see better days. At that time, my less experient self did not know about the Hoeffding inequality and the statistical test. I could have proposed removing HATR from skmultiflow and River, but I had a feeling I could maybe solve this puzzle given enough time.

Thus, I did my best in these past years to achieve that.

### 2.3. My first attempts

As I mentioned before, we monitor some type of error to detect drifts. In regression, a usual approach is to rely on absolute errors. Other things could also work, but the point is that these metrics are usually unbounded. The range of the error depends on the input data.

Since I did not know by then that the test used in the HAT classifier and replicated in HATR assumed Bernoulli variables, my first thought was to normalize the error between \\(0\\) and \\(1\\) and use the existing code.

I've tried different stuff, and Jacob Montiel always kindly listened to my crazy ideas and helped me with them. I tried using an incremental version of min-max scaling, but it generated a lot of false alarms. My second proposal powered HATR for years. 

My idea was to assume that the absolute errors passed to the drift detectors followed a normal distribution. If that were the case, I could push things further and use the empirical rule. This rule says that \\(99.73\%\\) of normally distributed data is between the interval defined by three times the standard deviation (\\(3 \times s\\)) around the mean value (\\(\overline{x}\\)). Note that I'm using the sample mean and standard deviation.


Using this interval, \\(\left[\overline{x} - 3s, \overline{x} + 3s\right]\\), we can normalize the errors using a more robust strategy. This strategy indeed can give us fewer false positive drift detections and was used for a long time in River. However, the statistical test assumptions were incorrect. Once I understood that everything made sense!

### 2.4. The current solution

Recently I discovered everything I talked about so far concerning HAT and HATR. Once I got all this matter about performing a statistical test to assess whether the differences between the performances of the foreground and background subtrees were significant, it was only a matter of finding something more appropriate for regression.

After some research, I decided to give a z-test a try. Why z-test? Well, it sounded generic and straightforward enough. Again, we assume the distributions are normally distributed (which sounds reasonable since we are monitoring errors and there are plenty of observations). By default, the threshold to define the warm-up period before testing for statistically significant performance differences is \\(300\\). This value comes from the original HAT. So, three hundred error observations for each contender (foreground and background subtrees) sounds reasonable.

And that's it. No more range limitations, mysterious formulae/code, and try/excepts!

For completeness, here is the [code](https://github.com/online-ml/river/blob/6f390b51b25f87989b463e11bd0460a9bf83e069/river/tree/nodes/hatr_nodes.py#L187-L200) for the performance difference significance in HATR:


```python
                alt_mean_er = self._alternate_tree._error_tracker.mean.get()
                cur_mean_er = self._error_tracker.mean.get()


                # Variance of the prediction error
                alt_s2_er = self._alternate_tree._error_tracker.get()
                cur_s2_er = self._error_tracker.get()


                # Perform z-test to determine whether mean errors are significantly different
                z = (alt_mean_er - cur_mean_er) / math.sqrt(alt_s2_er / alt_n + cur_s2_er / cur_n)
                # We double the p-value due to the double-tailed test
                p_value = 2.0 * tree._norm_dist.cdf(-abs(z))


                # The mean errors are significantly different accordingly to the z-test
                if p_value <= tree.switch_significance:
                    # From here, we select the best subtree to keep
```

For some extra flavor, besides adding the z-test, I also made the significance level of the subtree switching tests a configurable parameter in both HAT versions. They were hardcoded to \\(\alpha=0.05\\) previously.

# 3. Wrapping up

During my Ph.D. I was lucky to meet wonderful people who helped to figure out many things and maybe contribute somehow to Online ML. Some things take time, and I am happy that many people make open-source tools like River and others better and better.

Finally I was able to put together my first blog post (is it a blog?). It is nice to try something in a more free form in comparison with a research paper. Who knows, maybe I will try it more often.

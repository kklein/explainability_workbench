# SHAP
Paper #1: https://arxiv.org/pdf/1705.07874.pdf

Paper #2 (tailored to trees): https://www.nature.com/articles/s42256-019-0138-9.epdf?shared_access_token=RCYPTVkiECUmc0CccSMgXtRgN0jAjWel9jnR3ZoTv0O81kV8DqPb2VXSseRmof0Pl8YSOZy4FHz5vMc3xsxcX6uT10EzEoWo7B-nZQAHJJvBYhQJTT1LnJmpsa48nlgUWrMkThFrEIvZstjQ7Xdc5g%3D%3D

SHAP is distinct from Shapley.

## Shapley

Shapley values are a concept from game theory revolving around credit assignment to individuals of a coalition of players in a collaborative game. Here the desire to quantify player $i$'s contribution to the payoff/outcome $v$ arises.

The Shapley value for player $i \in N$ wrt payoff $v$ is defined as follows:

$$ \psi_i(v) = \sum_{S \subset N \setminus \{i\}} {|N| \choose 1, |S|, n-|S|-1}^{-1} v(S \cup \{i\}) - v(S)$$

Translating one of our use cases into the game-theoretic framework could lead to the following correspondence:


| collaborative game     | prediction task                       |
|------------------------|---------------------------------------|
| player                 | feature                               |
| payoff for players $N$ | $\|\mu - f(x_i)\|$ for features $x_i$ |

where $\mu$ is the mean of the prediction over a dataset $X$ such that $x_i \in X$.

## SHAP

Paper #1 introduced the notion of SHAP - Shapley Additive Explanations. Colloquially two concepts are associated with SHAP - in contrast to Shapley values:
* the restriction of the payoff function to be linear/additive in its features/players
* concrete suggestions as to how to estimate the SHAP values

Akin to the formalism introduced by LIME, the authors suggest the following:
* $f$ represents the original model, $g$ an explanatory model
* Yet, $g$ operates in the space of 'simplified inputs' where $x' \in \{0, 1\}^M$
  * The simplified input can be thought of as either choosing a feature (group) or not
  * E.g. with image data, $M$ could be a number of patches/super-pixels to either use in prediction or not.
  * With tabular data, we can usually assume that $x \ in R^M$ - in other words we have a one-to-one correspondence
  * A mapping function $h_x$ translates the simplified inputs to the inputs: $h_x(x') = x$ and $h_x(z') \approx z'$ if $z' \approx x$
  * The relation between $f$ and $g$ can be thought of as follows: $f(x) = g(x')$ where $h_x(x') = x$

With the help of these definitions, the authors suggest a restriction on the set of possible explanations. They argue that this restriction has implicitly been adopted by all major prior work as well:

$$ g(z') = \phi_0 + \sum_{i=1}^M z_i' \phi_i $$

where we refer to $\phi_i$ as the SHAP value of (simplified) feature $i$.

We observe that if $z' \neq \mathbb{1}_M$ then $h_x(z')$ will have absent values for some features/dimensions. Since many models are not able to predict with absent values, we will instead rewrite as follows:

$$ f(h_x(z')) = \mathbb{E} [f(z)|z_S] = \mathbb{E}_{z_{\tilde{S}} | z_S}[f(z)] $$

where $S$ indicates the set of original features selected by the simplified input and $\tilde{S}$ the remaining features.

Moreover, the authors suggest approximating the latter quantity by assuming:
1. Independence of non-selected features with respect to selected features, i.e. $z_S \perp z_{\tilde{S}}$
2. Linearity of the original function in its features

$$\begin{aligned}
f(h_x(z')) &= \mathbb{E}_{z_{\tilde{S}} | z_S}[f(z)] \\
	&= \mathbb{E}_{z_{\tilde{S}}}[f(z)] \\
	&= f(z_S, \mathbb{E}[z_{\tilde{S}}])
\end{aligned}$$

![](shap.png)

In practice, $\mathbb{E}[z_{\tilde{S}}]$ seems to be estimated either via its mean or via a random draw from another observation.

The authors argue that under these assumptions, LIME with a particular loss function, regularization and proximity measure can be equivalent to estimating SHAP values.

## TreeExplainer

While paper #1 introduced the model-agnostic KernelSHAP in order to estimate SHAP values, paper #2 introduces TreeExplainer, a SHAP value estimation approach tailored to decision trees and ensembles thereof. Most importantly, TreeExplainer distinguishes itself in two ways from KernelSHAP:
* Conceptually, it approximates $f(h_x(z'))$ differently
* Practically, it comes with an improvement on the asymptotic runtime bound

### Approximation

While KernelSHAP grants itself permission to assume $z_S \perp z_{\tilde{S}}$ and to therefore approximate $f(h_x(z'))$ based on the marginal expectation of $f(z)$ wrt $z_{\tilde{S}}$, TreeExplainer relies on a conditional expectation $\mathbb{E}_{z_tilde{S}|z_S}[f(z_tilde{S})|z_s]$.

On an intuitive level this conditional expectation can be approximated by computing a weighted average of all leaves that do not contradict the given feature values $z_S$. In other words, if node comes with a decision criterion based on feature $i$ and $i \in S$, we only follow the 'true' subtree. If, on the other hand, a node comes with a decision criterion based on feature $i$ and $i \notin S$, we average over both subtrees.

### Runtime

While KernelSHAP is said to have asymptotic runtime complexity of $\mathcal{O}(TLD^M)$, TreeExplainer is said to be bound by $\mathcal{O}(TLD^2)$ when applied to decision trees, where
* $T$ represents the number of trees
* $L$ represents the maximal number of leaves
* $D$ represents the maximal depth
* $M$ represents the number of features, as before

### Algorithm(s)
TODO.


## Notes

* Some report vulnerabilities to adversarial attacks: https://dl.acm.org/doi/pdf/10.1145/3375627.3375830
* The `shap` library provides different `Explainer` objects, such as
  * [explainers.Additive](https://shap.readthedocs.io/en/latest/generated/shap.explainers.Additive.html) (I would guess that this corresponds to KernelSHAP)
  * [explainers.Tree](https://shap.readthedocs.io/en/latest/generated/shap.explainers.Tree.html)
  * [explainers.LimeTabular](https://shap.readthedocs.io/en/latest/generated/shap.explainers.other.LimeTabular.html)

# SHAP
Paper #1: https://arxiv.org/pdf/1705.07874.pdf
Paper #2 (tailored to trees): https://arxiv.org/abs/1905.04610

## Shapley
SHAP is distinct from Shapley.

Shapley values are a concept from game theory revolving around credit assignment to individuals of a coalition of players in a collaborative game. Here the desire to quantify player $i$'s contribution to the payoff/outcome $v$ arises.

The Shapley value for player $i \in N$ wrt payoff $v$ is defined as follows:

$$ \psi_i(v) = \sum_{S \subset N \setminus \{i\}} {|N| \choose 1, |S|, n-|S|-1}^{-1} v(S \cup \{i\}) - v(S)$$

Translating one of our use cases into the game-theoretic framework could lead to the following correspondence:


| collaborative game     | prediction task                       |
|------------------------|---------------------------------------|
| player                 | feature                               |
| payoff for players $N$ | $\|\mu - f(x_i)\|$ for features $x_i$ |

where $\mu$ is the mean of the prediction over a dataset $X$ such that $x_i \in X$.

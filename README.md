# EXAM: Experiment-as-Market

A python package for Experiment-as-Market (EXAM), an experiment design algorithm which:

1. produces a Pareto efficient allocation of treatment assignment probabilities;

2. is asymptotically incentive compatible for preference elicitation;

3. unbiasedly estimates any causal effect estimable with standard RCTs.

Please refer to the [<a href="https://github.com/zliu392/exam-project/blob/master/documents/pseudocode.pdf">pseudocode</a>] for the details of EXAM algorithm. 

### Author(s)

This package is written and maintained by Yusuke Narita (yusuke.narita@yale.edu).


### Installation

The latest release of the package can be installed with pip:

```R
pip install experimentasmarket
```

Any published release can also be installed from source:

```R
pip install git+https://github.com/zliu392/exam-project
```

### Usage Examples

The following example demonstrates how to apply EXAM in experiment design applications.

```R
import experimentasmarket
```

### References
Yusuke Narita.
<b>Toward an Ethical Experiment</b>, available at
[<a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3094905">SSRN</a>].

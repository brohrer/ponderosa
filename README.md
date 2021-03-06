# WE'VE MOVED

### This repository is now being hosted on GitLab's servers.
Check it out at [https://gitlab.com/brohrer/ponderosa](https://gitlab.com/brohrer/ponderosa).
Install from there. Update from there.

To make the switch on your local machine command line run

git remote set-url origin git@gitlab.com:brohrer/ponderosa.git

or

git remote set-url origin https://gitlab.com/brohrer/ponderosa.git

depending on which protocol you're using.

This repo is being deprecated and will no longer be updated. BTW GitLab is pretty intuitive if you're already familiar with GitHub. [Check it out.](https://gitlab.com/)

--

--

--

--

--

--

# Pondersa, a discrete optimizer

Ponderosa is a companion for machine learning algorithms with higher
level parameters and design decisions that aren't automatically
optimized during the model fitting process.

It's a descrete optimizer, which means that it works with parameters that can
only take on a certain set of values. This works well for evaluating neural
networks, since it takes so long to train and test one.


![Evolutionary Powell's method, animated](/ponderosa/landing_page_demo.gif)

It includes Evolutionary Powell's method (pictured in action above), an evolutionary
search algorithm variant inspired by
[Powell's method](https://en.wikipedia.org/wiki/Powell%27s_method).
I believe Evolutionary Powell's method is novel. (Please let me know if
you've seen something like it before.)

## Installation

Whether you want to use Ponderosa or experiment with it, you'll want
to clone the repository to your local machine and install it from there.

```bash
git clone https://github.com/brohrer/ponderosa.git
python3 -m pip install -e ponderosa
```

## Try it out

```bash
python3
```
```python3
>>> import ponderosa.demo
```

## About the name

![Ponderosa Pine Tree, NPS photo by W. Kaesler](https://www.nps.gov/romo/learn/nature/images/ponderosa-tree-Walt-Kaesle_1.jpg?maxwidth=650&autorotate=false)

[Ponderosa Pine Tree, NPS photo by W. Kaesler](https://www.nps.gov/romo/learn/nature/conifers.htm)

The Ponderosa Pine is the tallest tree in the mountain forests I grew up hiking in.
Compared to the Scrub Oak and Sagebrush is was gigantic. For better or for worse,
this is appropriate to the size of hyperparameter optimization compute jobs.
They are ponderous.

Also, if you bury your nose in the rifts of bark, you can smell vanilla.
No relevance to optimization, but I always loved that.

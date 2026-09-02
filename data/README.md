# data/

Drop `.txt` files here to define your **workload** corpus, then run any tool
with `--split workload`.

This matters more than it looks. Redundancy is a property of a model *on a
distribution*. Measuring it on wikitext and deploying on code, chat, or your
own domain is exactly the calibration-set overfitting that makes published
pruning numbers fail to reproduce. `held_out` (wikitext-2) is the neutral
reference; the number you should actually optimise against is the one measured
on text that looks like your traffic.

Contents are gitignored.

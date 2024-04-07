![image](https://github.com/joy-void-joy/alexplainable/assets/56257405/7db54cd9-ab71-4898-8fb6-83732476fc8e)

# Alexplainable: Constructing an ImageNet recognizer to be explainable

Alexplainable is an attempt at making a semi-transparent [ImageNet](https://www.image-net.org/) recognizer using [AI-Pull-Requests](lesswrong post). Runs of this project will be had at [https://github.com/joy-void-joy/alexplainable-runs](https://github.com/joy-void-joy/alexplainable-runs).

Bet on [Manifold](https://manifold.markets/news/will-constructability-actually-work) about how much this process will succeed, and like us on [Manifund](manifund) to support our project.

General structure
===
The general purpose of this project is to investigate [constructability](lesswrong post): How feasible is it right now to construct AI projects from scratch in a way that makes it easy to understand what the different parts are.
We want to see how many classes of ImageNet we are able to construct a recognizer for, and how much human intervention does it require.

Our approach is the following:
- Define a graph of features using LLMs on images iteratively. Each node would be a feature that is directly observable and categorizable (like “bucket” or “rusty”)
- Segment the images, and classify the segments according to what nodes they activate.
- Train shallow convolutional neural networks on leaf nodes. When they have a good enough accuracy, have them commit a pull-request for integration
- Train convolutional neural network of ~1 layers for a feature at level (n+1) using only as input the feature-map of level n.
- If a node cannot be trained to have good enough accuracy in a fixed amount of time, use LLMs to refactor the graph further.

Early results
===
Code and results will be uploaded over the week, and this section updated.

Defining an ontology
---
Our process for defining an ontology has been to:
- Start from a pre-defined ontology made in 30 minute by hand
- Have claude describe an image thoroughly
- Extract a graph from the description
- Have it merge the current graph with the extracted graph.
 
Doing this, we have been able to extract from 10 images [an ontology that seems to make sense](link to ontology):

![image](https://github.com/joy-void-joy/alexplainable/assets/56257405/e3dfda63-f3ed-4abd-adbd-d9710ab38e12)

Some extract of them seems pretty solid and exactly like what we want:
```
│   ├── temporary fence
│   │   └── signs posted
│   │       ├── construction company information
│   │       └── safety warnings
```
Which is exactly what we want: a set of reviewable features that clearly compose together in a clear fashion.

This loop has two main problems however: some node needs to be cleaned up, like  “dense” or “timestamp”, and it currently grows linearly in the number of images:

![image](https://github.com/joy-void-joy/alexplainable/assets/56257405/38187ac5-259b-4933-9c14-0977db6a5a54)

We are still testing way to have Claude or other system refactor the ontology. For instance, it may be possible that using [function calls](https://docs.anthropic.com/claude/docs/tool-use) with Claude will yield better results than it did with GPT-4, and that this would make the ontology more succint.

Segmenting
---
Segmenting seems to work very well with [segment anything](https://github.com/facebookresearch/segment-anything):

![image](https://github.com/joy-void-joy/alexplainable/assets/56257405/7c652491-6cc5-4e09-a70b-614d66f3c8d7)

Classification of the segments with LLaVA would be cheap, but has been mixed. Claude however has been able to classify segments fairly well:

![image](https://github.com/joy-void-joy/alexplainable/assets/56257405/070ad9c2-9bff-4808-b44b-b77f8557bdb4)

As claude is costly, we expect to do some hand classification for now to have the datasets available for CNN-training

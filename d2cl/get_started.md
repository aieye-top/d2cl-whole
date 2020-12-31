# Getting Started

Please first install my favorite package `torch`.

```{.python .input  n=1}
!pip install torch
```

This course teaches full-stack production deep learning:[1]

- Formulating the problem and estimating project cost
- Finding, cleaning, labeling, and augmenting data
- Picking the right framework and compute infrastructure
- Troubleshooting training and ensuring reproducibility
- Deploying the model at scale

Who is this for

- The course is aimed at people who already know the basics of deep learning and want to understand the rest of the process of creating production deep learning systems. You will get the most out of this course if you have:
- At least one-year experience programming in Python.
- At least one deep learning course (at a university or online).
- Experience with code versioning, Unix environments, and software engineering.

Phase 1 is Project Planning and Project Setup: At this phase, we want to decide the problem to work on, determine the requirements and goals, as well as figure out how to allocate resources properly.
Phase 2 is Data Collection and Data Labeling: At this phase, we want to collect training data (images, text, tabular, etc.) and potentially annotate them with ground truth, depending on the specific sources where they come from.
Phase 3 is Model Training and Model Debugging: At this phase, we want to implement baseline models quickly, find and reproduce state-of-the-art methods for the problem domain, debug our implementation, and improve the model performance for specific tasks.
Phase 4 is Model Deployment and Model Testing: At this phase, we want to pilot the model in a constrained environment, write tests to prevent regressions, and roll the model into production.

[1]: https://course.fullstackdeeplearning.com/

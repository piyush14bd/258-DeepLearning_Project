This paper proposes an alternative to the standard Multi-Layer Perceptrons (MLPs) we use in deep learning model.

In standard MLP we have learnable weights on the edges and fixed activation functions like ReLU or Sigmoid on the nodes. KANs changes this. They get rid of the linear weights and place learnable activation functions (parametrized as B-splines) directly on the edges. The nodes then sum everything up.

Significance of the change: 

Parameter Efficiency: KANs can achieve better accuracy with way fewer parameters than MLPs. The paper shows them beating MLPs on scientific tasks with much smaller models.

Interpretability: The edges are univariate functions and we can visually inspect them. This brings more transparency to the black box nature of neural networks. We can actually see the mathematical formula the network has learned (e.g. seeing that an edge learned a sin wave).

The Trade-off 

KANs are slower to train than MLPs. MLPs are fast because they rely on matrix multiplication, which GPUs are heavily optimized for. KANs calculate individual spline functions that are currently about 10 times slower to train. I see this as an engineering bottleneck that needs to be optimized before it can replace MLPs at scale.
# p-DynSys_EncoderTransformerDecoder
Parametrised Uncertainty Aware ROM for Dynamical Systems

This is a variational and parametrised equivelent of DynSys_EncodeTranbsformerDecoder. We refere the reader to the article Parametrised and uncertainty-aware dynamic reduced-order model – application to unsteady flows. 
The expected data should have dimension [p, t, d + p.dim]
Where : 
  - p is the number of distinct parameter sets (parameter dimension). parameters refere to external variables that can trigger different responses from the model. The dimension of the parameter space is refered to as p.dim
  - t are the snapchots (time dimension).
  - d is the spatial dimension (in 1D).

Considering a Navier Stokes emulator, given data from a flow domain defined by a 100*100 grid, on the span of 1500 snapshots, this flow is given if 5 different configurations where a single variable has been modified such as the Reynolds Number. 
The dataset dimension should be [5, 1500, 10001]. The parameter value (here Reynolds number) is stacked at the bottom of the spatial vector. It is unique for each of the 5 [t, d + p.dim] sub-dataset. 


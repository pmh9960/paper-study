# InverseForm: A Loss Function for Structured Boundary-Aware Segmentation

- **Author:** Shubhankar Borse et al.
- **Institution:** Qualcomm AI Research
- **Published:** CVPR 2021 (Oral)
- **Arxiv:** https://arxiv.org/abs/2104.02745

## Contribution

1. Proposed boundary distance-based measure, InverseForm. 
   - Significantly more capable of capturing the spatial boundary transform than cross-entropy based beasure.
2. The scheme is agnostic to the backbone architecture choice.
   - Plug-and-play property.
   - Can fit into multi-task learning frameworks.
3. SOTA in both single-task (on NYU-Depth-v2), and multi-task settings (on PASCAL).

![Fig1](./fig1.png)

Figure 1. Left: Images from Cityscapes val benchmark. Middle: Segmented prediction for an HRNet-48-OCR baseline. Right: Same backbone trained using our InverseForm boundary loss.

## 3. Proposed Scheme: InverseForm

## 3.1. Motivation for distance-based metrics

![Fig2](./fig2.png)

Figure 2. Cross-entropy(XE) based distance fails for spatial transformations of boundaries.

- Pixel-wise cross-entropy or balanced cross-entropy losses take into account the pixel-wise features (intensity, etc.)
- but not spatial distance between object boundaries and ground-truth boundaries.
- They are insufficient for imposing boundary alignment for segmentation. (Ilustrated in Figure 2.)
- Accordingly, boundary detection networks trained with pixel-based losses produce thicker and distorted boundaries.
- Some works use Hausdorff distance to model this measure between boundaries,
- but this loss cannot be efficiently applied in a semantic segmentation settings.

## 3.2. Inverse transformation network

![Fig3](./fig3.png)

Figure 3. Spatial transformer (a) and our inversetransformation network (b).

- Assume that two boundary maps are related to each other through a homography transformation.
- Need to build Spatial Transformer Network. In this paper, $\theta$ is $3 \times 3$ matrix. (STN [[21]][STN], Figure 3(a).)
- Create a network that inputs two boundary maps and predicts the "homography change" as its output. (Figure 3(b).)
- This network is called inverse transformation network, because it performs the inverse operation of STN, theoretically.
- The outputs of the inverse transformation network are the coefficients of the homography matrix.
- There are numerous methods to formulate a distance metric from these values. In this paper, two distance metrices are choosed.
- One may also attempt to directly regress on the distance instead of estimating the transformation coefficients.
- However, such an approach would not allow optimization of the boundary-aware segmentation network. **(?)**

## 3.3. Measuring distances from homography

- If there is a perfect match between input boundary maps, the network should estimate an identity matrix.
- **Euclidean distance:** 
  - Train the inverse-transformation network by reducing $d_{if}(x, t_\theta (x)) = \left\Vert \hat{\theta} - \theta \right\Vert_F$.
  - At inference time, $d_{if}(x, t_\theta (x)) = \left\Vert \hat{\theta} - I_3 \right\Vert_F$.
- **Geodesic distance:** *(not sure)*
  - Homography transformations reside on an analytical manifold instead of a flat Euclidean space. 
  - Figure 1: The deviation between two transformations should be measured along the curved manifold (Lie group) of transformations rather than through the forbidden Euclidean space of transformations.
    ![euclidean_vs_geodesic](./euclidean_vs_geodesic.png)
    $$d_{if}(x, t_\theta (x)) = \left\Vert \frac{\text{Log}(\theta^{-1}\hat{\theta})}{\text{Log}(I_3)} \right\Vert_F$$
  - Above equation need Riemannian logarithm to calculate gradient, which does not have a closed-form solution.
  - In [[27]][AETv2], project the homography Lie group onto a subgroup $SO(3)$ [[41]][SO(3)] where the calculation of geodesic distance does not need the Riemannian logarithm.
  - Now, the formulation is given by,
    $$d_{if}(x, t_\theta (x)) = \arccos \left[ \frac{\text{Tr}(P) - 1}{2} \right] + \lambda \text{Tr} (R_\pi^T R_\pi)$$
    - Weighting parameter: $\lambda = 0.1$.
    - Projection $P$ onto the rotation group $SO(3)$: $P = U diag\{1, 1, det(UV)^T\} V^T$
    - Projection residual: $R_\pi = \theta^{-1} \hat{\theta} - P$
  - During inference, $\theta = I_3$

## 3.4. Using InverseForm as a loss function

- At first, train the inverse-transformation network using boundary maps of images sampled from the target dataset.
- Apply the STN [[21]][STN] to generate the transformed versions of boundary images. It leads greater realistic transformations than randomly sampling transformation parameters.
- Before feeding boundary maps to the network, images are split into smaller tiles.
- Ideally, the best tiling dimension should provide a balance between local and global contexts. (The effect of tiling dimension in the Appendix.)
- Assume the predicted boundary $b_{pred}$ is a transformed version of the ground truth boundary label $b_{gt}$. i.e. $b_{pred} = t_\theta(b_{gt})$.
  $$L_{if}(b_{pred}, b_{gt}) = \sum_{j=1}^N d_{if}(b_{pred, j}, b_{gt, j})$$

## 3.5. Boundary-aware segmentation setup

![Fig4](./fig4.png)

Figure 4. Overall framework for our proposed boundary-aware segmentation.

- Single-task architectures using InverseForm loss.
- Use a simple boundary-aware segmentation setup. (Figure 4.)
- This setup could be used over any backbone.
  $$L_{total} = L_{xe}(y_{pred}, y_{gt}) + \beta L_{bxe} (b_{pred}, b_{gt}) + \gamma L_{if}(b_{pred}, b_{gt})$$

## 4. Experimental Results

## 4.1. Results on NYU-Depth-v2

![Table1](./table1.png)

![Table2](./table1.png)

![Fig5](./fig5.png)

![Fig7](./fig7.png)

## 4.2. Results on PASCAL

![Table3](./table3.png)

## 4.3. Results on Cityscapes

![Fig6](./fig6.png)

![Table4](./table4.png)

## 5. Ablation Studies

- **Searching for the best inverse-transformation:**
  - Compared to the convolutional architecture used in AET [[52]][AET]
    ![Table5](./table5.png)
- **Distance function:**
  - There is no clear winner.
  - Geodesic distance can lead to exploding gradients easily. This severely limits the search-space for hyperparameters.
  - Euclidean distance might not model perspective homography best, but has wider search-space and hence a more consistent improvement.

<!-- Reference -->
[STN]: https://proceedings.neurips.cc/paper/2015/file/33ceb07bf4eeb3da587e268d663aba1a-Paper.pdf
[AET]: https://arxiv.org/abs/1901.04596
[AETv2]: https://arxiv.org/abs/1911.07004
[SO(3)]: https://www.cis.upenn.edu/~cjtaylor/PUBLICATIONS/pdfs/TaylorTR94b.pdf
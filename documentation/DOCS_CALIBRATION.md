# calibration.py 

>**Overview:** `NeedlePivotCalibrator` calculates the static 3D offset between a tracking sensor **(Rigid Body)** and the functional needle tip. It using least-squares optimization to ensure sub-millimeter navigation accuracy. 

## Functions 

| Function Name | Input | Output | Logic Summary |
| :--- | :--- | :--- | :--- |
| `add_pose` | `R, t, quality` | `None` | Standardizes and stores the incoming tracking samples (Rotation Matrix **R** and Position Vector **t**). | 
| `has_sufficient_diversity` | `None` | `bool` | Makes sure the user has moved the needle at least 10 degrees to prevent mathematical errors. |
| `_stack_system` | `mask` | `(A, b)` | Builds the linear system Ax = b for the pivot model. | 
| `_least_squares` | `mask` |` x (vector)` | Initial guess. | 
| `solve` | `None` | `Dict` | Runs RANSAC for outlier rejection. Then runs Huber-loss for optimization. | 
| `save_json`, `load_json` | `path` | `None/Self` | Store calibrated offsets to disk. | 
| `apply_tip` | `T, r_tip` | `3x1 vec` | Applies the calibrated offset to a live-tracking pose. | 

## Additional Information 

| Function Name | Input | Output | Logic Summary |
| :--- | :--- | :--- | :--- |
| `_project_to_so3` | `3x3 mat ` | `3x3 mat` | Raw tracking data sometimes warps due to noise. Use Singular Value Decomposition **(SVD)** for orthogonalization to return the closest rotation matrix and ensure a mathematically valid rotation is still present. | 
| `_rotation_angle` | `R_a, R_b` | `float` | Returns the rotation angle in radians between two rotation matrices using the **Trace**. Used in the "diversity check" **(has_sufficient_diversity)** to prevent bad calibrations by checking angular spread. | 

## Mathematical Justification 

### Least Squares and Root Mean Square: 

$$\text{Minimize: } \sum_{i=1}^{n} (r_i)^2$$

$$
RMS = \sqrt{\frac{1}{n} \sum \|r\|^2}
$$

where $r = Ax - b$, the distance (error) between where the model says the tip should be and where the sensor says it actually is for each frame. For an $RMS > 1mm$, needle tip recalibration is required. 

### Singular Value Decomposition: 

Ideally, the rotation matrix ***R** is orthogonal: $R^TR = I$ and $det(R) = 1$. Noise skews the matrix. SVD is used to find the closest valid rotation matrix. 

Any *mxn* matrix M can be factored into 3 distinct matrices:

$$
M =U\Sigma V^T
$$ 

where $U$ is an *mxm* orthogonal matrix (for rotation in output space), $\Sigma$ is an *mxn* diagonal matrix representing scaling or stretching, and $V^T$ is an *nxn* orthogonal matrix (for rotation in input space). 

The $\Sigma$ is responsible for the warped rotation. Diagonal values would be 1.0 if $M$ were a perfect rotation. 

$$
M = U \begin{bmatrix} \sigma_1 & 0 & 0 \\ 0 & \sigma_2 & 0 \\ 0 & 0 & \sigma_3 \end{bmatrix} V^T
$$

To get the closest rotation matrix, replace $\Sigma$ with the Identity Matrix $I$ after matrix decomposition. 

$$
R = U I V^T = U V^T
$$

Afterwards, we need to ensure that the coordinate system is not mirrored. Check: 

$$ 
det(R) = 1
$$ 

If $det(R) = -1$ multiply the last column of $U$ by -1. 

$$
R_{fixed} = [u_1, u_2, -u_3] V^T
$$

### RANSAC: 

Random Sample Consensus is an iterative trial-and-error method to find the best data points within a given dataset containing inliers and outliers. Used to minimize effect of outliers. Inliers are determined based on error threshold manually set. 

### Huber-Loss Optimization: 

Huber Loss is a "hybrid" function that changes behavior based on the size of the error. Applied after RANSAC, when an inlier set of data is determined. Error size is determined based on a threshold manually set. 

| error size | penalty type | behavior 
| :--- | :--- | :--- |
| `small` | `quadratic` | L2: $Loss = \frac{1}{2}(error)^2$ | 
| `large` | `linear` | L1: $Loss = \delta(\lvert error \rvert - \frac{1}{2}\delta)$ | 


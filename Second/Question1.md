### 1. 代数证明题

**证明：在一元线性回归中，残差之和为 0，且残差与自变量乘积之和为 0。**
设一元线性回归模型为：$ y_i = \beta_0 + \beta_1 x_i + e_i $
使用最小二乘法，目标是最小化残差平方和（RSS）：
$$
Q(\hat{\beta}_0, \hat{\beta}_1) = \sum_{i=1}^n e_i^2 = \sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2
$$
分别对 $ \hat{\beta}_0 和\hat{\beta}_1 $ 求偏导，并令偏导数为 0 以取得极值：
**1) 对 $ \hat{\beta}_0 $ 求偏导：**
$$
\frac{\partial Q}{\partial \hat{\beta}_0} = -2 \sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0
$$
两边同时除以 -2：
$$
\sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0
$$
因为 $ e_i = y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i\ $，所以得到结论：
$$
\sum_{i=1}^n e_i = 0
$$
*(得证：残差之和必然等于零)*
**2) 对 $ \hat{\beta}_1\ $ 求偏导：**
$$
\frac{\partial Q}{\partial \hat{\beta}_1} = -2 \sum_{i=1}^n x_i(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0
$$
两边同时除以 -2，并代入 $e_i\ $：
$$
\sum_{i=1}^n x_i e_i = 0
$$
*(得证：残差与自变量乘积之和也必然等于零)*
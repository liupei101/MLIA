## KNN model

### 算法流程

输入：有标记的数据T

输出：实例x输出其所属类别y

算法：寻找实例中与x，距离度量最近的k个点，这k个点中类别占比最多的作为实例x的类别。

### 关键点

**距离度量**：计算空间中距离的方式；

**k值**：k越小，模型的复杂度越高，近似误差越小，但是学习的估计误差很大；

**分类决策**：一般为取类别占比最多进行类别归属决策。属于经验风险最小化误差。

### 优化

**KD-Tree**：一种类似于二叉树的数据结构，树上按照某种规则存储训练样本，可以高效的搜索与检索数据。
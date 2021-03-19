# OTVAE
我们的国创项目
The repository of our National College Students Innovation Program.

[项目成果论文-最优传输理论与其几何基础
Essay-Optimal Transport and its Geometric Basis](https://github.com/jimcui3/OTVAE/blob/main/%E7%A0%94%E7%A9%B6%E6%88%90%E6%9E%9C-%E6%9C%80%E4%BC%98%E4%BC%A0%E8%BE%93%E7%90%86%E8%AE%BA%E4%B8%8E%E5%85%B6%E5%87%A0%E4%BD%95%E5%9F%BA%E7%A1%80.pdf)

[项目结果论文-生成对抗网络与其几何解释
Essay-Generative Adversarial Network and its Geometric Basis](https://github.com/jimcui3/OTVAE/blob/main/%E7%A0%94%E7%A9%B6%E6%88%90%E6%9E%9C-%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%E4%B8%8E%E5%85%B6%E5%87%A0%E4%BD%95%E8%A7%A3%E9%87%8A.pdf)

[结项报告-生成对抗网络的几何解释及算法优化
Final report-Geometric View and Algorithm Optimization of Generative Adversarial Network (GAN)](https://github.com/jimcui3/OTVAE/blob/main/%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%E7%9A%84%E5%87%A0%E4%BD%95%E8%A7%A3%E9%87%8A%E5%8F%8A%E7%AE%97%E6%B3%95%E4%BC%98%E5%8C%96%20%E9%A1%B9%E7%9B%AE%E7%BB%93%E9%A1%B9%E6%8A%A5%E5%91%8A.pdf)

[项目所用的WGAN代码与生成模型OTVAE的代码
WGAN and OTVAE codes](https://github.com/jimcui3/OTVAE/tree/main/OTVAE)

其中WGAN.py可以直接运行。OTVAE需要将OTVAE.py，optimal_transport.py，par.pth放在同一路径下。OTVAE.py为主文件，可以在其最后一行进行参数设置。par.pth为VAE训练100个epoch时得到的参数权重，可以直接进行读取，在epoch为100时，OTVAE的优秀率约为85-90%。

WGAN.py can run directly. You need to put OTVAE.py, optimal_transport.py, and par.pth under the same path to run OTVAE. OTVAE.py is the main file, its parameters can be set in its last line. par.pth is the weight obtained when VAE is trained for 100 epochs, and can be directly read. The excellence rate of OTVAE is about 85-90% after training for 100 epoches.

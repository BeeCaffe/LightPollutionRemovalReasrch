## 光污染消除项目研究
本项目主要研究沉浸式投影系统中光线互反射造成的投影图片质量下降的问题。
### 项目结构
```
--checkpoint //存放相关训练结果的权重、
--output     //网络补偿结果的一些输出
--res        //一些其它图片资源
--src        //存放源码
  -- cyclegan  //cyclegan源码
  -- pairnet   //基于注意力机制的双网络结构补偿网络
  --unet       //单补偿网络结构，形式像unet一样
  --tools      //一些工具函数
```
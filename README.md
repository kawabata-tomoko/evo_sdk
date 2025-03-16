# EVO-SDK v1.01

## Introduction

这个存储库是基于EVO的存储库进行的分叉，对部分代码进行了重构以适应下游任务的开发。此外对EVO中的代码结构进行了简单的分区和整理，将不同功能的代码切分开来以减少阅读难度。

**请注意：本存储库并非原生EVO模型，向模型stateless_forward方法添加了gradient_checkpoint的支持以及embedding的获取（通过config设置config.unembedding=False获取，否则返回的是logits值）

该存储库的目标是为EVO模型的微调工作提供助力，用户可以简单的开发部署潜在的下游任务模型。

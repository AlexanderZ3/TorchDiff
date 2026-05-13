这是一个任务说明文档， 我需要完成这样的任务：
1. 要开源两个训练好的权重和一个训练recipes  分别到cann-recipes-infer仓库和 cann-recipes-train仓库。 目前两个仓库我已经下载好了， 本地地址分别是 D:\Doc\OSP\cann-recipes-train-master  和 D:\Doc\OSP\cann-recipes-infer-master；
2， 对于在train仓开源要开源的，是D:\Doc\OSP\TorchDiff  这个项目中关于NPU系列的osp_next模型的14B的，关于skiparse这一稀疏化attention的训练操作， 这是基于wan2.1基座模型的， 要训的话大家就直接基于wan2.1——t2v_14B  的权重训练即可。 同时train仓其实已经有几个经典模型适配进去了， 现在入仓之前，  必须写一个类似于D:\Doc\OSP\cann-recipes-train-master\llm_rl\deepseek\README.md之类的readme 文档， 来介绍自己的模型训练， 以及怎么训练， 需要哪些硬件等等， 你可以分析参考一下那个文件； 同时， 之前我写过一个 D:\Doc\OSP\TorchDiff\CANN仓开源文档\OSP_Next_14B_BF16_README.md， 其中有些信息是可以参考的。 你现在就基于 D:\Doc\OSP\TorchDiff这个仓库中的npu osp_next  skiparse这一条训练track， 写一个类似的md文档。 

3. 对于infer仓库要开源的， 也是D:\Doc\OSP\TorchDiff  这个项目中关于NPU系列的osp_next模型的14B的， 只不过是有两个权重， 一个是bf16正常训好的， 一个是使用hif8训练过的，我们在这个文档中要点明两个模型权重都要开源， 且要说清楚两个模型怎么使用，哪里下载，需要多少配置等等， 这些一部分信息你可以参考D:\Doc\OSP\TorchDiff\CANN仓开源文档\OSP_Next_14B_HiF8_README.md，  然后格式问题你可以参考cann-recipes-infer-master仓库中本身就有的很多个模型， 类似于D:\Doc\OSP\cann-recipes-infer-master\models\wan2.2-i2v这个模型， 这是和我们最接近的。 

4， 目前对于适配两个仓库的代码， 暂时先不用修改，我当前最急迫的就是写好两个介绍文档， 告诉别人我要开源什么，要怎么用我开源的东西， 你一定要把框架搭建好， 至于某些信息缺失，比如权重下载链接还暂时没有， 这些都可以空着，但要点明。  同时，这两个仓库中有contribution.md文档告诉开发者应该怎么贡献代码进入这个仓

5，帮我输出两个这样的文档， 到D:\Doc\OSP\TorchDiff\CANN仓开源文档




TRAIN:
**4. 后续入仓还缺哪些内容？**

需要补齐样例代码目录、依赖锁定、可公开权重链接、数据准备说明、优化文档、性能/显存数据、License 说明，并按社区流程提交 RFC Issue 与 PR。

INFER:
**4. 入仓前还缺什么？**

需要补齐权重下载链接、Docker 镜像、依赖锁定、性能数据、优化文档，以及按 `cann-recipes-infer/CONTRIBUTION.md` 提交 RFC Issue 与 PR。
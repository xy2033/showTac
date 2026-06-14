该readme用于告诉学长如何运行这个项目的训练和推理

项目执行目录:/showO/Show-o/show-o2

训练脚本可以任选bash脚本：show-o2/run_tactile_stage_one.sh 或者 py脚本：show-o2/run_tactile_inference.py 来执行
训练时需要替换的事情：

conda环境
从 https://huggingface.co/datasets/xuyang2003/data_indoor的conda目录下载,然后上传到集群上

在集群的anaconda/env下创建文件夹并解压

cd ~/anaconda/env
mkdir showO
cd showO
sudo tar -zxvf show.tar.gz
这样就成功创建了showO conda 环境，也就是项目需要的环境
（可以参考https://blog.csdn.net/baidu_35692628/article/details/136519579?fromshare=blogdetail&sharetype=blogdetail&sharerId=136519579&sharerefer=PC&sharesource=xy2003king&sharefrom=from_link的方法一）

需要通过conda 环境来替换 run_tactile_stage_one.sh的 
PYTHON_EXECUTABLE和ACCELERATE_LAUNCH_MODULE的/root/miniconda3/envs/showO
请在集群上进行conda env list ,把对应showO环境的目录给替换上去

数据集
https://huggingface.co/datasets/xuyang2003/data_indoor目录下的data_indoor.zip就是数据集。
请进入到数据集包含3dprint，3dpirnt2多个物体子文件夹的目录，用这个目录的路径来替换run_tactile_stage_one.sh的TACTILE_DATA_ROOT=/defaultShare/data_indoor

模型
https://huggingface.co/xuyang2003/tac_showO 包含了所有需要的模型
请创建一个models文件夹，来包含https://huggingface.co/xuyang2003/tac_showO 下的所有模型（4个）
并且用models文件夹的目录来替换
run_tactile_stage_one.sh的MODEL_ROOT=/defaultShare/models

GPU训练个数控制
通过run_tactile_stage_one.sh的export CUDA_VISIBLE_DEVICES=0,1
来控制GPU个数

结果文件在show-o2/wandb 和 show-o2/outputs 学长把这两部分返回给我就行了

推理脚本可以任选bash脚本：show-o2/run_tactile_inference.sh 或者 py脚本：show-o2/run_tactile_inference.py 来执行

需要替换conda ,数据集和模型 和 训练替换方法一样 （替换对象是下面这四个变量）
PYTHON_EXECUTABLE="/root/miniconda3/envs/showO/bin/python"
ACCELERATE_LAUNCH_MODULE="/root/miniconda3/envs/showO/lib/python3.10/site-packages/accelerate/commands/launch.py
MODEL_ROOT=/defaultShare/models
TACTILE_DATA_ROOT=/defaultShare/data_indoor

推理额外需要替换的是训练输出的 checkpoint
STAGE1_CHECKPOINT=/Show-o/show-o2/outputs/showo2-1.5b-tactile-stage-1_video/checkpoint-45000/unwrapped_model
请替换为训练时保存的checkpoint

推理的结果在show-o2/Inference
学长把这个返回给我就行了



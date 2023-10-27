相对较好模型超参数

| 模型          | 训练阶段  | 使用数据                                                                                                                                                          | prefix                             | 对话能力 | 超参                                                                                                         | checkpoint steps | 卡数         | 评测效果        |
| ------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------ | ---------------- | ------------ | --------------- |
| bloom-820m-zh | 全参数SFT | <p>1、firefly-1.1M（共24类，每一类选500样本）<p>2、自我介绍28条，18条正面，10条否定（“你是谁”“你是不是ChatGPT”）                                                  | 你是由XXX开发的XXX，被设计来生成…… | 单轮对话 | <p>1、ZeRO-stage2<p>2、lr=5e-5,per_device_batch_size=8,accumulation_steps=1,scheduler=cosine, optim=adamw_hf | 7000-10000       | 32G-V100 * 1 | 远低于ChatGLM   |
| ^             | ^         | ^                                                                                                                                                                 | -                                  | 单轮对话 | <p>1、ZeRO-stage2<p>2、lr=5e-5,per_device_batch_size=8,accumulation_steps=1,scheduler=cosine, optim=adamw_hf | 7000-10000       | 32G-V100 * 1 | -               |
| baichuan-7B   | 全参数SFT | <p>1、firefly-1.1M（共24类，每一类选500样本）<p>2、自我介绍28条，18条正面，10条否定（“你是谁”“你是不是ChatGPT”）                                                  | 你是由XXX开发的XXX，被设计来生成…… | 单轮对话 | <p>1、ZeRO-stage2<p>2、lr=5e-5,per_device_batch_size=8,accumulation_steps=1,scheduler=cosine, optim=adamw_hf | 7000-10000       | 80G-A800 * 8 | 低于ChatGLM-1   |
| ^             | ^         | <p>1、firefly-1.1M（共24类，每一类选500样本）<p>2、自我介绍28条，18条正面，10条否定（“你是谁”“你是不是ChatGPT”）<p>3、alpaca_gpt4_zh 4w                           | -                                  | 单轮对话 | <p>1、ZeRO-stage2<p>2、lr=5e-5,per_device_batch_size=8,accumulation_steps=1,scheduler=cosine, optim=adamw_hf | 50000            | 80G-A800 * 8 | 略低于ChatGLM-1 |
| ^             | ^         | <p>1、firefly-1.1M（共24类，每一类选500样本）<p>2、自我介绍28条，18条正面，10条否定（“你是谁”“你是不是ChatGPT”）<p>3、alpaca_gpt4_zh 4w<p>4、belle multiturn 0.8M | -                                  | 多轮对话 | <p>1、ZeRO-stage2<p>2、lr=5e-5,per_device_batch_size=8,accumulation_steps=1,scheduler=cosine, optim=adamw_hf | 50000            | 80G-A800 * 8 | -               |
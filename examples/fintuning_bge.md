# fintuning bge model

## rerievel model

**install the package**
`pip install -U FlagEmbedding`

**data format**
`{"query": str, "pos": List[str], "neg":List[str]}`
**参数说明**
`query`是查询，`pos`是肯定文本列表，`neg`是否定文本列表。如果查询没有否定文本，您可以从整个语料库中随机采样一些文本作为否定文本。

**参考微调数据集**: [toy_finetune_data.jsonl](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/finetune/toy_finetune_data.jsonl)

**挖掘困难负样本**
```
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path BAAI/bge-large-zh-v1.5 \
--input_file your.jsonl \
--output_file your_finetune_data_minedHN.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--use_gpu_for_searching 
```
**参数说明**
`input_file`：用于微调的 json 数据。该脚本将为每个查询检索 top-k 文档，并从 top-k 文档中随机抽取负样本（不包括正文档）。
`output_file`：保存带有挖掘的硬底片以进行微调的 JSON 数据的路径
`negative_number`：采样负数的数量
`range_for_sampling`：在哪里采样负数。例如，2-100表示negative_number从 top2-top200 文档中采样负数。您可以设置更大的值来降低负数的难度（例如，设置60-300为从top60-300段落中采样负数）
`candidate_pool`：要检索的池。默认值为 None，此脚本将从neg中所有内容的组合中检索input_file。该文件的格式与预训练数据相同。如果输入候选池，该脚本将从该文件中检索negatives。
`use_gpu_for_searching`：是否使用 faiss-gpu 检索negatives。

```
torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-large-zh-v1.5 \
--train_data ./your_data.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval "" 
```
**参数设置**:
`per_device_train_batch_size`：训练中的批量大小。在大多数情况下，更大的批量大小会带来更强的性能。您可以通过启用--fp16、--deepspeed ./df_config.json（df_config.json 可以参考ds_config.json）--gradient_checkpointing等来扩展它。
`train_group_size`：训练中查询的positive and negatives的数量。总是有一个positive，因此该参数将控制negatives的数量 (#negatives=train_group_size-1)。请注意，negatives的数量不应大于 data 中negatives的数量"neg":List[str]。
`learning_rate`：对于大型/基础/小型，推荐 1e-5/2e-5/3e-5。
`temperature`：会影响相似度分数的分布。
`query_max_len`：查询的最大长度。请根据您数据中查询的平均长度进行设置。
`passage_max_len`：请根据您的数据中段落的平均长度进行设置。
`query_instruction_for_retrieval`：查询指令，会添加到每个查询中。您还可以将其设置""为不添加任何指令。
`use_inbatch_neg`：使用同一批中的段落作为`negatives`。默认值为 True。

## reranking model
可以使用和上面相同的数据
```
torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.reranker.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-reranker-base \
--train_data ./toy_finetune_data.jsonl \
--learning_rate 6e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--train_group_size 16 \
--max_len 512 \
--weight_decay 0.01 \
--logging_steps 10
```




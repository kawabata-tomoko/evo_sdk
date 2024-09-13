import torch
from evo.srcs.Application.SeqClsForEvo import SeqClsForEvo
from transformers import AutoConfig,AutoTokenizer
from transformers import DefaultDataCollator,Trainer,TrainingArguments
from datasets import load_dataset, load_from_disk
import wandb 
import os

os.environ["WANDB_PROJECT"]="evo_cas12_finetning"
wandb.login()
# logger = logging.get_logger(__name__)
torch.manual_seed(42)
configs=AutoConfig.from_pretrained(
    "/home/zhengyulong/models/project/configs",
    trust_remote_code=True,
    use_cache=False,
    num_labels=2
    )

model = SeqClsForEvo.from_pretrained(
    "/home/zhengyulong/models/project/models/pretrained/evo-1-131base",
    config=configs,
    torch_dtype=torch.float16
)

tokenizer=AutoTokenizer.from_pretrained(
    "/home/zhengyulong/models/project/configs/tokenizer/",
    trust_remote_code=True,
    cls_token="@",
    eos_token="&",
    bos_token="^",
    pad_token = 'N'
    )

datacollator = DefaultDataCollator()

training_args=TrainingArguments(
    output_dir="/home/zhengyulong/models/project/models/finetuning/16scls_freeze_finetuning",
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=20,
    learning_rate= 5e-6,#EVO use 0.00009698,
    lr_scheduler_type= "cosine",
    warmup_ratio = 0.01,
    weight_decay=0.05,#EVO use 0.1
    num_train_epochs=5,#EVO use 10
    gradient_accumulation_steps=1,#pretrained 8
    per_device_train_batch_size=16,#pretrained 4
    per_device_eval_batch_size=16,#pretrained 4
    bf16=True,
    logging_steps =5,
    report_to="wandb"
)
training_args=TrainingArguments(
    output_dir="/home/zhengyulong/models/project/models/finetuning/16scls_freeze_finetuning",
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=20,
    learning_rate= 5e-6,#EVO use 0.00009698,
    lr_scheduler_type= "cosine",
    warmup_ratio = 0.01,
    weight_decay=0.05,#EVO use 0.1
    num_train_epochs=5,#EVO use 10
    gradient_accumulation_steps=1,#pretrained 8
    per_device_train_batch_size=16,#pretrained 4
    per_device_eval_batch_size=4,#pretrained 4
    bf16=True,
    logging_steps =5,
    report_to="wandb"
)
# def pack(_tokenizer,max_length,padding="max_length",pad_to_multiple_of=None,return_tensors="pt"):
#     def padseq(line):
#         inputs=_tokenizer(list(map(lambda x :x+_tokenizer.eos_token,line["seq"])))
#         eos_index=list(map(lambda x:[len(x)-1],inputs["input_ids"]))
#         input_ids_padded=_tokenizer.pad(
#                 inputs,
#                 padding=padding,
#                 max_length=max_length,
#                 pad_to_multiple_of=pad_to_multiple_of,
#                 return_tensors=return_tensors
#                 )
#         return dict(
#             input_ids=input_ids_padded["input_ids"],
#             attention_mask=input_ids_padded["attention_mask"],
#             label=line["label"],
#             eos_index=eos_index
#         )
#     return padseq


# train_ds = load_dataset("csv",data_files="/home/zhengyulong/models/project/datasets/16s_cls.csv")
# # train_ds = load_dataset("json",data_files="/home/zhengyulong/models/project/datasets/dataset.json")
# func=pack(tokenizer,2000,padding="max_length")
# train_ds_sp=train_ds.map(
#     func,batched=True,num_proc=4)["train"]
# train_ds_sp=train_ds_sp.remove_columns("seq")
# tempset=train_ds_sp.train_test_split(test_size=0.4,seed=42)
# trainset=tempset["train"]
# tempset=tempset["test"]
# tempset=tempset.train_test_split(test_size=0.5,seed=42)
# evalset=tempset["train"]
# testset=tempset["test"]
# trainset.save_to_disk("/home/zhengyulong/models/project/datasets/cls_16s/trainset", num_proc=os.cpu_count())
# evalset.save_to_disk("/home/zhengyulong/models/project/datasets/cls_16s/evalset", num_proc=os.cpu_count())
# testset.save_to_disk("/home/zhengyulong/models/project/datasets/cls_16s/testset", num_proc=os.cpu_count())

trainset=load_from_disk("/home/zhengyulong/models/project/datasets/cls_16s/trainset")
evalset=load_from_disk("/home/zhengyulong/models/project/datasets/cls_16s/evalset")

def p_count(m):
    ttp=0
    tp=0
    for p in m.parameters():
        c=p.numel()
        if p.requires_grad == True:
            ttp+=c
        tp+=c
    print(f"Total trainable parameters: {ttp}")
    print(f"Total parameters: {tp}")

# print("before",model.backbone.embedding_layer.weight)
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
for p in model.backbone.parameters():
    p.requires_grad = False

for p in model.backbone.blocks[-1].parameters():
    p.requires_grad = True
# for p in model.backbone.embedding_layer.parameters():
#     p.requires_grad = True

p_count(model)
print(model.hidden.weight)
trainer= Trainer(
    model=model,
    args=training_args,
    train_dataset= trainset,
    eval_dataset= evalset,
    data_collator=datacollator,
)

trainer.train(resume_from_checkpoint="/home/zhengyulong/models/project/models/finetuning/16scls_freeze_finetuning/checkpoint-900")

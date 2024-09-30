import torch
from evo.srcs.Application.SeqClsForEvo import SeqClsForEvo
from transformers import AutoConfig,AutoTokenizer
from transformers import DefaultDataCollator,Trainer,TrainingArguments
from datasets import Dataset, load_dataset
import os
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    modules_to_save=["hidden","classifier"],
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
)

torch.manual_seed(42)
configs=AutoConfig.from_pretrained(
    "evo/configs",
    trust_remote_code=True,
    use_cache=False,
    num_labels=2
    )

model = SeqClsForEvo.from_pretrained(
    "$YOUR_MODEL_PATH$/models/pretrained/evo-1-131base",
    config=configs,
    torch_dtype=torch.float16
)



model.add_adapter(lora_config,adapter_name="translate_eff")
tokenizer=AutoTokenizer.from_pretrained(
    "evo/configs/tokenizer/",
    trust_remote_code=True,
    cls_token="@",
    eos_token="&",
    bos_token="^",
    pad_token = 'N'
    )

datacollator = DefaultDataCollator()

training_args=TrainingArguments(
    output_dir="${OUTPUT PATH}$",
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=50,
    save_total_limit=20,
    learning_rate= 5e-6,#EVO use 0.00009698,
    lr_scheduler_type= "cosine",
    warmup_ratio = 0.1,
    weight_decay=0.05,#EVO use 0.1
    num_train_epochs=5,#EVO use 10
    gradient_accumulation_steps=4,#pretrained 8
    per_device_train_batch_size=2,#pretrained 4
    per_device_eval_batch_size=4,#pretrained 4
    neftune_noise_alpha=10.0,
    max_grad_norm=10,
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

# train_ds = load_dataset("json",data_files="$WORK_DIR/datasets/sars/sars-sars2.json")

# func=pack(tokenizer,30500,padding="max_length")
# train_ds_sp=train_ds.map(
#     func,batched=True,num_proc=4)["train"]
# train_ds_sp=train_ds_sp.remove_columns("seq")
# tempset=train_ds_sp.train_test_split(test_size=0.2,seed=42)
# trainset=tempset["train"]
# tempset=tempset["test"]
# tempset=tempset.train_test_split(test_size=0.5,seed=42)
# evalset=tempset["train"]
# testset=tempset["test"]
# trainset.save_to_disk("$WORK_DIR/datasets/SARS-SARS2/trainset", num_proc=os.cpu_count())
# evalset.save_to_disk( "$WORK_DIR/datasets/SARS-SARS2/evalset", num_proc=1)
# testset.save_to_disk( "$WORK_DIR/datasets/SARS-SARS2/testset", num_proc=1)# from datasets import load_dataset, load_from_disk
from datasets import load_dataset, load_from_disk
trainset=load_from_disk("$WORK_DIR/datasets/SARS-SARS2/trainset")
evalset=load_from_disk("$WORK_DIR/datasets/SARS-SARS2/evalset")

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


p_count(model)
print(model.hidden.weight)

from torch import nn 
class StepFrozenTrainer(Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.epoch=0
        # self.layers = [
        #     module 
        #     for module in self.model.backbone.blocks.modules()
        #     if isinstance(module,nn.Module)
        #     ]
        # self.pass_epoch=[]
    def training_step(self, model, inputs):# 在每个 epoch 开始时调用 _freeze_layersif self.state.epoch != self.epoch:
        self.epoch = self.state.epoch
        # self._freeze_layers()
        return super().training_step(model, inputs)
    # def _freeze_layers(self):
    #     if int(self.epoch) not in self.pass_epoch:
    #         print("switched paras")
    #         self.pass_epoch.append(int(self.epoch))
    #         for layer in self.layers:
    #             for parm in layer.parameters():
    #                 parm.requires_grad = False
    #         for p in self.layers[-1*(int(self.epoch)+1)].parameters():
    #             p.requires_grad = True
# trainer= Trainer(
#     model=model,
#     args=training_args,
#     train_dataset= trainset,
#     eval_dataset= evalset,
#     data_collator=datacollator,
# )
import numpy as np
from sklearn.metrics import roc_auc_score
def compute_metrics(p):
    logits, labels = p
    pred=np.argmax(logits, axis=2).T[0]
    TP=np.sum((pred==1)&(labels==1))
    FP=np.sum((pred==1)&(labels==0))
    FN=np.sum((pred==0)&(labels==1))
    TN=np.sum((pred==0)&(labels==0))
    precision=TP/(FP+TP)
    recall=TP/(FN+TP)
    roc=roc_auc_score(
        labels,
        logits[:,:,1].T[0]
        )
    return {"precision":precision,"recall":recall,"roc-auc":roc}
trainer= StepFrozenTrainer(
    model=model,
    args=training_args,
    train_dataset= trainset,
    eval_dataset= evalset,
    data_collator=datacollator,
    compute_metrics=compute_metrics
)
trainer.train(resume_from_checkpoint="$WORK_DIR/models/finetuning/SARScls/checkpoint-1350")

import torch
from evo_sdk import SeqClsForEvo
from transformers import AutoConfig,AutoTokenizer
from transformers import DefaultDataCollator,Trainer,TrainingArguments

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
torch.manual_seed(42)
configs=AutoConfig.from_pretrained(
    "evo_sdk/configs",
    trust_remote_code=True,
    use_cache=False,
    num_labels=2
    )

model = SeqClsForEvo.from_pretrained(
    "$YOUR_MODEL_PATH$/models/pretrained/evo-1-131base",
    config=configs,
    torch_dtype=torch.float16
)

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

trainer= Trainer(
    model=model,
    args=training_args,
    train_dataset= trainset,
    eval_dataset= evalset,
    data_collator=datacollator,
    compute_metrics=compute_metrics
)
trainer.train(resume_from_checkpoint="$WORK_DIR/models/finetuning/SARScls/checkpoint-1350")

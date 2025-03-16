from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import logging

from evo_sdk.StripedHyenaPreTrainedModel import StripedHyenaPreTrainedModel
from evo_sdk.model import StripedHyena,print_rank_0
from evo_sdk.utils import dotdict

logger = logging.get_logger(__name__)


class SeqClsForEvo(StripedHyenaPreTrainedModel):
    supports_gradient_checkpointing=True
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.backbone = StripedHyena(dotdict(config.to_dict()))
        self.backbone.gradient_checkpointing = False
        self.config = config
        vocab_size = config.vocab_size
        if vocab_size % config.make_vocab_size_divisible_by != 0:
            vocab_size += config.make_vocab_size_divisible_by - (
                vocab_size % config.make_vocab_size_divisible_by
            )

        self.vocab_size = vocab_size
        self.num_labels = config.num_labels
        self.hidden = torch.nn.Linear(config.hidden_size,config.hidden_size*2,dtype=torch.float32)#.to(torch.bfloat16)
        self.classifier = torch.nn.Linear(config.hidden_size*2,self.num_labels,dtype=torch.float32)#.to(torch.bfloat16)#load as bf16
        self.ln_hidden = torch.nn.LayerNorm(config.hidden_size*2,dtype=torch.float32)
        self.post_init()
        self.force_dtype()
        
        
    def force_dtype(self):
        self.backbone.to_bfloat16_except_poles_residues() 
        
    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.backbone.gradient_checkpointing = enable
        super()._set_gradient_checkpointing(enable, gradient_checkpointing_func)

    def get_input_embeddings(self):
        return self.backbone.embedding_layer
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values=None,
        return_dict: Optional[bool] = None,
        eos_index : Optional[bool] = None 
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # eos_index = eos_index if eos_index is not None else torch.ones(input_ids.shape[0],1,dtype=int)*input_ids.shape[1]-1
        hidden_state, inference_params_dict = self.backbone(
            input_ids,
            inference_params_dict=past_key_values if use_cache else None,
            padding_mask=attention_mask
            )
        # feature=logits[:,-1,:] #use [EOS] Instead [CLS]
        # print_rank_0(hidden_state.shape)
        logits = self.classifier(self.ln_hidden(F.gelu(self.hidden(hidden_state))))

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        # eos_index=eos_index.to(hidden_state.device)
        # hidden_state = hidden_state.to(dtype=self.hidden.weight.dtype).gather(1, eos_index.unsqueeze(-1).expand(-1, -1, hidden_state.size(-1)))
        # logits = self.classifier(self.ln_hidden(F.gelu(self.hidden(hidden_state))))
        
        loss = None
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()#ignoring label:-100
            labels = labels.to(pooled_logits.device)
            loss = loss_fct(pooled_logits.view(-1,self.num_labels), labels)
            
        if return_dict:
            return SequenceClassifierOutput(
                loss = loss,
                logits = pooled_logits,
                hidden_states = None,#hidden_state,
                attentions = None
                )
        else:
            return pooled_logits

    @classmethod
    def can_generate(cls) -> bool:
        return False

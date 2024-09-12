import torch
from torch import nn
from project.srcs.BaseModel.skelecton.model import StripedHyena
from project.srcs.BaseModel import StripedHyenaPreTrainedModel
from project.utils.utils import dotdict
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import logging
from typing import Optional, Tuple, Union
import torch.nn.init as init
logger = logging.get_logger(__name__)

class EmbedStripedHyena(StripedHyena):
    def __init__(self,contig):
        super().__init__(contig)
    def forward(self, x, inference_params_dict=None, padding_mask=None):
        L = x.shape[1]
        x = self.embedding_layer.embed(x)
        if inference_params_dict is not None:
            x, inference_params_dict_out = self.stateful_forward(
                x,
                inference_params_dict=inference_params_dict,
            )
        else:
            x, inference_params_dict_out = self.stateless_forward(x, padding_mask=padding_mask)

        x = self.norm(x)
        # x = self.unembed.unembed(x)
        return x, inference_params_dict_out
class SeqClsForEvo(StripedHyenaPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        model_config = dotdict(config.to_dict())
        self.backbone = EmbedStripedHyena(model_config)
        self.backbone.gradient_checkpointing = False
        self.config = config
        vocab_size = config.vocab_size
        if vocab_size % config.make_vocab_size_divisible_by != 0:
            vocab_size += config.make_vocab_size_divisible_by - (
                vocab_size % config.make_vocab_size_divisible_by
            )

        self.vocab_size = vocab_size
        self.num_labels = config.num_labels
        self.hidden = torch.nn.Linear(config.hidden_size,config.hidden_size*2).to(torch.bfloat16)
        self.classifier = torch.nn.Linear(config.hidden_size*2,self.num_labels).to(torch.bfloat16)#load as bf16
        self.ln_hidden = torch.nn.LayerNorm(config.hidden_size*2,dtype=torch.bfloat16)
        self.post_init()
        self.force_dtype()
        
        
    def force_dtype(self):
        self.backbone.to_bfloat16_except_poles_residues() 
        
    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.backbone.gradient_checkpointing = enable

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
        eos_index = eos_index if eos_index is not None else torch.ones(input_ids.shape[0],1,dtype=int)*input_ids.shape[1]-1
        
        if use_cache:
            if self.backbone.gradient_checkpointing and self.backbone.training:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
            elif labels is not None:
                logger.warning_once(
                    "`use_cache=True` is incompatible with loss calculation. Setting `use_cache=False`..."
                )
                use_cache = False

        logits, past_key_values = self.backbone(
            input_ids,
            padding_mask=attention_mask,
            inference_params_dict=past_key_values if use_cache else None,
        )

        # feature=logits[:,-1,:] #use [EOS] Instead [CLS]
        eos_index=eos_index.to(logits.device)
        feature = logits.gather(1, eos_index.unsqueeze(-1).expand(-1, -1, logits.size(-1)))

        # feature.to(self.hidden.weight.dtype)
        feature = self.ln_hidden(torch.tanh(self.hidden(feature)))
        logits = torch.nn.functional.softmax(self.classifier(feature),dim=2)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()#ignoring label:-100

            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1,self.num_labels), labels)

        if return_dict:
            return SequenceClassifierOutput(
                loss = loss,
                logits = logits,
                hidden_states = None,
                attentions = None
                )
        else:
            return logits

    @classmethod
    def can_generate(cls) -> bool:
        return False

import pytest
import torch
from algorithm.model.bert_torch import BertForSst2Torch

def test_bert_torch():
    model = BertForSst2Torch(from_pretrained=False, num_labels=2)
    assert model.bert is not None
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones((2, 16), dtype=torch.long)
    token_type_ids = torch.zeros((2, 16), dtype=torch.long)
    labels = torch.tensor([0, 1])
    
    loss, logits, prob = model(input_ids, attention_mask, token_type_ids, labels)
    assert loss is not None
    assert logits.shape == (2, 2)
    assert prob.shape == (2, 2)
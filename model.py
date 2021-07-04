# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/1 21:39
@Author  : SinGaln
"""
from Rotransformer import RoTransformerEncoder
from transformers import BertModel, BertPreTrainedModel

class BertModelOutputs(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertModelOutputs, self).__init__(config)
        self.args = args
        self.BertOutput = BertModel(config=config)

        self.ro_transformer = RoTransformerEncoder(args)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.BertOutput(input_ids=input_ids, token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        sequence_outputs = outputs[0]
        pooled_outputs = outputs[1]
        logits = self.ro_transformer(sequence_outputs)
        return logits

# if __name__=="__main__":
#     parser = argparse.ArgumentParser()
#     """配置参数测试"""
#     parser.add_argument("--embedding_size", default=768, type=int,required=True, help="The hidden size of model.")
#     parser.add_argument("--hidden_size", default=1024, type=int,required=True, help="The hidden size of model.")
#     parser.add_argument("--num_attention_heads", default=12, type=int, required=True, help="The number of attention heads.")
#     parser.add_argument("--attention_dropout_prob", default=0.2, type=float, required=True, help="The dropout rate of attention.")
#     parser.add_argument("--feed_dropout_rate", default=0.1, type=float, required=True, help="The dropout rate of attention.")
#
#     args = parser.parse_args()
#     model = RoTransformerEncoder(args)
#     a = model(torch.rand(32, 512, 768))
#     print(a.shape)
import torch
import transformers

class Classifier(transformers.BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		print(config)
		self.bert = transformers.BertModel(config)
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.clf_head = torch.nn.Linear(config.hidden_size, config.num_labels)

		self.init_weights()

	def forward(self, input_ids,
			**kwargs):
		encoder_output = self.bert(input_ids)
		pooled_output = encoder_output.last_hidden_state[:,0,:]
# 		pooled_output = encoder_output[1]
# 		pooled_output = self.dropout(pooled_output)
		clf_logit = self.clf_head(pooled_output)
		return transformers.modeling_outputs.SequenceClassifierOutput(
				logits=clf_logit,
				hidden_states=encoder_output,
				)

	def generate(self, input_ids,
			**kwargs):
		clf_logit = self.forward(input_ids).logits
		clf_index = torch.max(clf_logit, dim=1)[1]
		return clf_index

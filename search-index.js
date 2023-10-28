var searchIndex = JSON.parse('{\
"candle_lora":{"doc":"","t":"NIIIIDDDDDDDDDDIEGDNDDLLLLKKKLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLKKLLMMLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLMKLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLKLLLLKLLLLLLLLLLLLLLLLLLLLLLLLLLLLLMKLLLLLLLLLLLLLLKLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLKLLLLLLLLLLLLLLLLLLKKKLLL","n":["AlreadyMerged","Conv1dLayerLike","Conv2dLayerLike","EmbeddingLayerLike","LinearLayerLike","Lora","LoraConfig","LoraConv1d","LoraConv1dConfig","LoraConv2d","LoraConv2dConfig","LoraEmbedding","LoraEmbeddingConfig","LoraLinear","LoraLinearConfig","Merge","MergeError","MergeErrorOrError","NewLayers","NotMerged","SelectedLayers","SelectedLayersBuilder","add_conv1d_layers","add_conv2d_layers","add_embed_layers","add_linear_layers","bias","bias","bias","bias","bias","bias","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","build","clone","clone","clone","clone","clone","clone","clone","clone","clone","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","config","config","config","config","conv1d","conv2d","convert_model","default","deref","deref","deref","deref","deref","deref","deref","deref","deref","deref","deref","deref","deref","deref","deref_mut","deref_mut","deref_mut","deref_mut","deref_mut","deref_mut","deref_mut","deref_mut","deref_mut","deref_mut","deref_mut","deref_mut","deref_mut","deref_mut","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","embed","embeddings","embeddings","eq","equivalent","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","forward","forward","forward","forward","from","from","from","from","from","from","from","from","from","from","from","from","from","from","get_delta_weight","get_delta_weight","get_delta_weight","get_delta_weight","get_delta_weight","hidden_size","hidden_size","init","init","init","init","init","init","init","init","init","init","init","init","init","init","into","into","into","into","into","into","into","into","into","into","into","into","into","into","linear","merge_weights","merge_weights","merge_weights","merge_weights","merge_weights","new","new","new","new","new","new","new","new","new","new","shape","shape","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_string","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","unmerge_weights","unmerge_weights","unmerge_weights","unmerge_weights","unmerge_weights","vzip","vzip","vzip","vzip","vzip","vzip","vzip","vzip","vzip","vzip","vzip","vzip","vzip","vzip","weight","weight","weight","weight","weight","weight"],"q":[[0,"candle_lora"]],"d":["","Any layer that is conv1d-like.","Any layer that is conv2d-like.","Any layer that is embedding-like.","Any layer that is linear-like.","","","","Configuration for LoraConv1d. Other configurations are …","","Configuration for LoraConv2d. Other configurations are …","","Configuration for LoraEmbedding, with <code>num_embeddings</code> …","","Configuration for LoraLinear","","","","New layers, after conversion","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Convert the selected layers into their LoRA counterparts.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Get the delta weight of the LoRA layer. This is meant to …","","","","","","","","","","","","","","","","","","","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","Merge the LoRA weights.","","","","","","","","","","","","","","Create a new LoRA config.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Unmerge the LoRA weights.","","","","","","","","","","","","","","","","","","","","","","","",""],"i":[27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,27,0,0,4,4,4,4,12,5,8,16,17,18,39,19,4,25,16,7,17,9,20,11,18,13,21,27,39,19,4,25,16,7,17,9,20,11,18,13,21,27,4,16,7,17,9,20,11,18,13,21,16,7,17,9,20,11,18,13,21,5,8,16,17,25,25,39,4,39,19,4,25,16,7,17,9,20,11,18,13,21,27,39,19,4,25,16,7,17,9,20,11,18,13,21,27,39,19,4,25,16,7,17,9,20,11,18,13,21,27,25,10,20,27,27,16,7,17,9,20,11,18,13,21,27,27,16,17,20,18,39,19,4,25,16,7,17,9,20,11,18,13,21,27,40,16,17,20,18,10,20,39,19,4,25,16,7,17,9,20,11,18,13,21,27,39,19,4,25,16,7,17,9,20,11,18,13,21,27,25,40,16,17,20,18,4,16,7,17,9,20,11,18,13,21,12,18,16,7,17,9,20,11,18,13,21,27,39,19,4,25,16,7,17,9,20,11,18,13,21,27,39,19,4,25,16,7,17,9,20,11,18,13,21,27,39,19,4,25,16,7,17,9,20,11,18,13,21,27,40,16,17,20,18,39,19,4,25,16,7,17,9,20,11,18,13,21,27,12,5,8,16,17,18],"f":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[[[4,[[0,[1,2,3]]]],[6,[[0,[1,2,3]],5]],7],[[4,[[0,[1,2,3]]]]]],[[[4,[[0,[1,2,3]]]],[6,[[0,[1,2,3]],8]],9],[[4,[[0,[1,2,3]]]]]],[[[4,[[0,[1,2,3]]]],[6,[[0,[1,2,3]],10]],11],[[4,[[0,[1,2,3]]]]]],[[[4,[[0,[1,2,3]]]],[6,[[0,[1,2,3]],12]],13],[[4,[[0,[1,2,3]]]]]],[[],[[15,[14]]]],[[],[[15,[14]]]],[[],[[15,[14]]]],[16,[[15,[14]]]],[17,[[15,[14]]]],[18,[[15,[14]]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[[4,[[0,[1,2,3]]]]],[[19,[[0,[1,2,3]]]]]],[16,16],[7,7],[17,17],[9,9],[20,20],[11,11],[18,18],[13,13],[21,21],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[],22],[[],23],[16,22],[17,23],0,0,[[[19,[[0,[1,2,3]]]],21,24],[[25,[[0,[1,2,3]]]]]],[[],[[4,[[0,[1,2,3]]]]]],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],[26],0,[[],14],[20,14],[[27,27],28],[[],28],[[16,29],30],[[7,29],30],[[17,29],30],[[9,29],30],[[20,29],30],[[11,29],30],[[18,29],30],[[13,29],30],[[21,29],30],[[27,29],30],[[27,29],30],[[16,14],[[31,[14]]]],[[17,14],[[31,[14]]]],[[20,14],[[31,[14]]]],[[18,14],[[31,[14]]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[],[[33,[14,32]]]],[16,[[33,[14,32]]]],[17,[[33,[14,32]]]],[20,[[33,[14,32]]]],[18,[[33,[14,32]]]],[[],26],[20,26],[[],26],[[],26],[[],26],[[],26],[[],26],[[],26],[[],26],[[],26],[[],26],[[],26],[[],26],[[],26],[[],26],[[],26],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],0,[[],[[33,[32]]]],[16,[[33,[32]]]],[17,[[33,[32]]]],[20,[[33,[32]]]],[18,[[33,[32]]]],[[],[[4,[[0,[1,2,3]]]]]],[[5,7,21,24,26],[[31,[16]]]],[[26,26,26],7],[[8,9,21,24,26],[[31,[17]]]],[[26,26],9],[[10,11,21,24,26],[[31,[20]]]],[[26,26],11],[[12,13,21,24,26],[[31,[18]]]],[[26,26],13],[[26,34,[15,[35]]],21],[[],36],[18,36],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[],37],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],33],[[],38],[[],38],[[],38],[[],38],[[],38],[[],38],[[],38],[[],38],[[],38],[[],38],[[],38],[[],38],[[],38],[[],38],[[],[[33,[32]]]],[16,[[33,[32]]]],[17,[[33,[32]]]],[20,[[33,[32]]]],[18,[[33,[32]]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[],14],[[],14],[[],14],[16,14],[17,14],[18,14]],"c":[],"p":[[8,"Eq"],[8,"PartialEq"],[8,"Hash"],[3,"SelectedLayersBuilder"],[8,"Conv1dLayerLike"],[3,"HashMap"],[3,"LoraConv1dConfig"],[8,"Conv2dLayerLike"],[3,"LoraConv2dConfig"],[8,"EmbeddingLayerLike"],[3,"LoraEmbeddingConfig"],[8,"LinearLayerLike"],[3,"LoraLinearConfig"],[3,"Tensor"],[4,"Option"],[3,"LoraConv1d"],[3,"LoraConv2d"],[3,"LoraLinear"],[3,"SelectedLayers"],[3,"LoraEmbedding"],[3,"LoraConfig"],[3,"Conv1dConfig"],[3,"Conv2dConfig"],[6,"VarBuilder"],[3,"NewLayers"],[15,"usize"],[4,"MergeError"],[15,"bool"],[3,"Formatter"],[6,"Result"],[6,"Result"],[6,"MergeErrorOrError"],[4,"Result"],[15,"f64"],[15,"f32"],[3,"Shape"],[3,"String"],[3,"TypeId"],[3,"Lora"],[8,"Merge"]]},\
"candle_lora_transformers":{"doc":"","t":"AAAAAAAAADDDRDLLLLLLLLLLLLLLLLLLLLMLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLDLLLLLLLLLLLLLLFLDDMMMLLLLMLLLLLLLLMLLLLLLLLMMLLMLLMLMMMMMMLLLLLLMLMLLDDDDDRLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLMMLLLLLMMLLLLLLLLMMMMMMMMMMLLLLLLLLLLLLLLLLLMMMMLLLLLLDDLLLLLLLLLLLLLLLLLLLLLMMLLMLLMLMMMMMMLLLLLLLMMLLDDLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLDDDLLLLLLLLLLLLLLLLLLLLLLLLLMLLLLLLLLLLLLLLLLLLMLLLLLLLLLLMLLLFFFDDDDLLLLLLLLLLLLFLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLFFLLLLLLLLLLLLLLLLLLLL","n":["bert","dinov2","falcon","llama","mistral","stable_lm","t5","varbuilder_utils","with_tracing","BertLinear","BertModel","Config","DTYPE","LayerNorm","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","clone","clone_into","default","deref","deref","deref","deref","deref_mut","deref_mut","deref_mut","deref_mut","deserialize","device","drop","drop","drop","drop","eq","fmt","fmt","fmt","forward","forward","forward","from","from","from","from","get_lora_model","get_merged_lora_model","init","init","init","init","into","into","into","into","load","new","new","to_owned","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","vzip","vzip","vzip","vzip","DinoVisionTransformer","borrow","borrow_mut","deref","deref_mut","drop","fmt","forward","from","init","into","new","try_from","try_into","type_id","vit_small","vzip","Config","Falcon","alibi","attention_dropout","bias","borrow","borrow","borrow_mut","borrow_mut","bos_token_id","config","default","deref","deref","deref_mut","deref_mut","drop","drop","eos_token_id","falcon7b","fmt","fmt","forward","from","from","get_lora_model","get_merged_lora_model","hidden_dropout","hidden_size","init","init","initializer_range","into","into","layer_norm_epsilon","load","multi_query","n_head_kv","new_decoder_architecture","num_attention_heads","num_hidden_layers","parallel_attn","try_from","try_from","try_into","try_into","type_id","type_id","use_cache","validate","vocab_size","vzip","vzip","Cache","Config","Llama","LlamaConfig","LlamaLinear","MAX_SEQ_LEN","bias","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","clone","clone_into","config_7b_v1","config_7b_v2","deref","deref","deref","deref","deref","deref_mut","deref_mut","deref_mut","deref_mut","deref_mut","deserialize","drop","drop","drop","drop","drop","fmt","forward","forward","from","from","from","from","from","get_lora_model","get_lora_model","get_merged_lora_model","get_merged_lora_model","hidden_size","hidden_size","init","init","init","init","init","intermediate_size","intermediate_size","into","into","into","into","into","into_config","load","new","num_attention_heads","num_attention_heads","num_hidden_layers","num_hidden_layers","num_key_value_heads","num_key_value_heads","rms_norm_eps","rms_norm_eps","rope_theta","rope_theta","shape","to_owned","try_from","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","type_id","use_flash_attn","use_kv_cache","vocab_size","vocab_size","vzip","vzip","vzip","vzip","vzip","weight","Config","Mistral","borrow","borrow","borrow_mut","borrow_mut","clone","clone_into","config_7b_v0_1","deref","deref","deref_mut","deref_mut","drop","drop","eq","fmt","fmt","forward","from","from","get_lora_model","get_merged_lora_model","hidden_act","hidden_size","init","init","intermediate_size","into","into","max_position_embeddings","new","num_attention_heads","num_hidden_layers","num_key_value_heads","rms_norm_eps","rope_theta","sliding_window","to_owned","try_from","try_from","try_into","try_into","type_id","type_id","use_flash_attn","vocab_size","vzip","vzip","Config","Model","borrow","borrow","borrow_mut","borrow_mut","clone","clone_into","deref","deref","deref_mut","deref_mut","drop","drop","eq","fmt","fmt","forward","from","from","head_dim","init","init","into","into","new","num_kv_groups","rotary_ndims","stablelm_3b_4e1t","to_owned","try_from","try_from","try_into","try_into","type_id","type_id","vzip","vzip","Config","T5EncoderModel","T5ForConditionalGeneration","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","clear_kv_cache","clear_kv_cache","clone","clone_into","decode","default","deref","deref","deref","deref_mut","deref_mut","deref_mut","deserialize","device","device","drop","drop","drop","encode","eos_token_id","eq","fmt","fmt","fmt","forward","forward","from","from","from","init","init","init","into","into","into","load","load","musicgen_small","pad_token_id","to_owned","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","use_cache","vzip","vzip","vzip","from_mmaped_safetensors","from_npz_tensors","from_pth_tensors","QMatMul","TracedLoraConv2d","TracedLoraEmbedding","TracedLoraLinear","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","clone","clone","clone_into","clone_into","conv2d","deref","deref","deref","deref","deref_mut","deref_mut","deref_mut","deref_mut","drop","drop","drop","drop","embeddings","fmt","fmt","fmt","fmt","forward","forward","forward","forward","from","from","from","from","from_weights","get_lora_model","get_lora_model","get_merged_lora_model","get_merged_lora_model","init","init","init","init","into","into","into","into","linear","linear_no_bias","new","new","to_owned","to_owned","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","vzip","vzip","vzip","vzip"],"q":[[0,"candle_lora_transformers"],[9,"candle_lora_transformers::bert"],[80,"candle_lora_transformers::dinov2"],[97,"candle_lora_transformers::falcon"],[150,"candle_lora_transformers::llama"],[253,"candle_lora_transformers::mistral"],[302,"candle_lora_transformers::stable_lm"],[340,"candle_lora_transformers::t5"],[402,"candle_lora_transformers::varbuilder_utils"],[405,"candle_lora_transformers::with_tracing"]],"d":["","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Be sure to provide a configuration for each type!","Be sure to provide a configuration for each type!","","","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Be sure to provide a configuration for each type!","Be sure to provide a configuration for each type!","","","","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","Load a Falcon model which will be converted to a LoRA …","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Be sure to provide a configuration for each type!","Be sure to provide a configuration for each type!","Be sure to provide a configuration for each type!","Be sure to provide a configuration for each type!","","","","","","","","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","Load a Mistral model which will be converted to a LoRA …","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Be sure to provide a configuration for each type!","Be sure to provide a configuration for each type!","","","","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","Load a Llama model which will be converted to a LoRA model.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","","Load tensors into a VarBuilder backed by a VarMap using …","Load tensors into a VarBuilder backed by a VarMap using …","Load tensors into a VarBuilder backed by a VarMap using …","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","","Be sure to provide a configuration for each type!","Be sure to provide a configuration for each type!","Be sure to provide a configuration for each type!","Be sure to provide a configuration for each type!","","","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","","","","",""],"i":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,6,9,1,10,6,9,1,1,1,1,10,6,9,1,10,6,9,1,1,10,10,6,9,1,1,6,9,1,10,6,9,10,6,9,1,6,6,10,6,9,1,10,6,9,1,10,6,9,1,10,6,9,1,10,6,9,1,10,6,9,1,10,6,9,1,0,22,22,22,22,22,22,22,22,22,22,22,22,22,22,0,22,0,0,24,24,24,24,23,24,23,24,23,24,24,23,24,23,24,23,24,24,24,23,23,24,23,23,23,24,24,24,23,24,24,23,24,23,24,24,24,24,24,24,24,23,24,23,24,23,24,24,24,24,23,0,0,0,0,0,0,25,27,28,25,26,29,27,28,25,26,29,26,26,27,27,27,28,25,26,29,27,28,25,26,29,28,27,28,25,26,29,25,25,29,27,28,25,26,29,25,29,25,29,27,28,27,28,25,26,29,27,28,27,28,25,26,29,28,29,26,27,28,27,28,27,28,27,28,27,28,25,26,27,28,25,26,29,27,28,25,26,29,27,28,25,26,29,27,26,27,28,27,28,25,26,29,25,0,0,33,34,33,34,33,33,33,33,34,33,34,33,34,33,33,34,34,33,34,34,34,33,33,33,34,33,33,34,33,34,33,33,33,33,33,33,33,33,34,33,34,33,34,33,33,33,34,0,0,35,36,35,36,35,35,35,36,35,36,35,36,35,35,36,36,35,36,35,35,36,35,36,36,35,35,35,35,35,36,35,36,35,36,35,36,0,0,0,39,37,38,39,37,38,37,38,39,39,38,39,39,37,38,39,37,38,39,37,38,39,37,38,38,39,39,39,37,38,37,38,39,37,38,39,37,38,39,37,38,37,38,39,39,39,39,37,38,39,37,38,39,37,38,39,39,37,38,0,0,0,0,0,0,0,50,51,47,48,50,51,47,48,47,48,47,48,0,50,51,47,48,50,51,47,48,50,51,47,48,50,50,51,47,48,50,51,47,48,50,51,47,48,51,50,51,50,51,50,51,47,48,50,51,47,48,0,0,50,48,47,48,50,51,47,48,50,51,47,48,50,51,47,48,50,51,47,48],"f":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[1,1],[[]],[[],1],[2],[2],[2],[2],[2],[2],[2],[2],[3,[[4,[1]]]],0,[2],[2],[2],[2],[[1,1],5],[[6,7],8],[[9,7],8],[[1,7],8],[[10,11,11],[[12,[11]]]],[[6,11],[[12,[11]]]],[[9,11],[[12,[11]]]],[[]],[[]],[[]],[[]],[[6,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],[[6,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],[[],2],[[],2],[[],2],[[],2],[[]],[[]],[[]],[[]],[[14,1,5,13],[[12,[10]]]],[[14,11,[16,[11]],5,13],6],[[11,11,20],9],[[]],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],21],[[],21],[[],21],[[],21],[[]],[[]],[[]],[[]],0,[[]],[[]],[2],[2],[2],[[22,7],8],[[22,11],[[12,[11]]]],[[]],[[],2],[[]],[[14,2,2,2,5,13],[[12,[22]]]],[[],4],[[],4],[[],21],[[14,5,13],[[12,[22]]]],[[]],0,0,0,0,0,[[]],[[]],[[]],[[]],0,[23,24],[[],24],[2],[2],[2],[2],[2],[2],0,[[],24],[[24,7],8],[[23,7],8],[[23,11],[[12,[11]]]],[[]],[[]],[[23,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],[[23,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],0,0,[[],2],[[],2],0,[[]],[[]],0,[[14,24,5,13,15,19],[[12,[23]]]],0,0,0,0,0,0,[[],4],[[],4],[[],4],[[],4],[[],21],[[],21],0,[24,12],0,[[]],[[]],0,0,0,0,0,0,[25,[[16,[11]]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[26,26],[[]],[5,27],[5,27],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[3,[[4,[28]]]],[2],[2],[2],[2],[2],[[25,7],8],[[25,11],[[12,[11]]]],[[29,11,2],[[12,[11]]]],[[]],[[]],[[]],[[]],[[]],[[25,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],[[29,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],[[25,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],[[29,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],0,0,[[],2],[[],2],[[],2],[[],2],[[],2],0,0,[[]],[[]],[[]],[[]],[[]],[[28,5],27],[[14,26,27,5,13,15,19],[[12,[29]]]],[[5,30,27,31],[[12,[26]]]],0,0,0,0,0,0,0,0,0,0,[25,32],[[]],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],21],[[],21],[[],21],[[],21],[[],21],0,0,0,0,[[]],[[]],[[]],[[]],[[]],[25,11],0,0,[[]],[[]],[[]],[[]],[33,33],[[]],[5,33],[2],[2],[2],[2],[2],[2],[[33,33],5],[[33,7],8],[[34,7],8],[[34,11,2],[[12,[11]]]],[[]],[[]],[[34,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],[[34,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],0,0,[[],2],[[],2],0,[[]],[[]],0,[[33,14,5,13],[[12,[34]]]],0,0,0,0,0,0,[[]],[[],4],[[],4],[[],4],[[],4],[[],21],[[],21],0,0,[[]],[[]],0,0,[[]],[[]],[[]],[[]],[35,35],[[]],[2],[2],[2],[2],[2],[2],[[35,35],5],[[35,7],8],[[36,7],8],[[36,11,2],[[12,[11]]]],[[]],[[]],[35,2],[[],2],[[],2],[[]],[[]],[[35,14,5,13],[[12,[36]]]],[35,2],[35,2],[5,35],[[]],[[],4],[[],4],[[],4],[[],4],[[],21],[[],21],[[]],[[]],0,0,0,[[]],[[]],[[]],[[]],[[]],[[]],[37],[38],[39,39],[[]],[[38,11,11],[[12,[11]]]],[[],39],[2],[2],[2],[2],[2],[2],[3,[[4,[39]]]],[37,31],[38,31],[2],[2],[2],[[38,11],[[12,[11]]]],0,[[39,39],5],[[39,7],8],[[37,7],8],[[38,7],8],[[37,11],[[12,[11]]]],[[38,11,11],[[12,[11]]]],[[]],[[]],[[]],[[],2],[[],2],[[],2],[[]],[[]],[[]],[[14,39,5,13],[[12,[37]]]],[[14,39,5,13],[[12,[38]]]],[[],39],0,[[]],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],21],[[],21],[[],21],0,[[]],[[]],[[]],[[[42,[[41,[40]]]],30,31],[[4,[[45,[[44,[43]]]],46]]]],[[[41,[40]],30,31],[[4,[[45,[[44,[43]]]],46]]]],[[[41,[40]],30,31],[[4,[[45,[[44,[43]]]],46]]]],0,0,0,0,[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[47,47],[48,48],[[]],[[]],[[2,2,2,49,14],[[12,[47]]]],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[50,11],[[50,7],8],[[51,7],8],[[47,7],8],[[48,7],8],[[50,11],[[12,[11]]]],[[51,11],[[12,[11]]]],[[47,11],[[12,[11]]]],[[48,11],[[12,[11]]]],[[]],[[]],[[]],[[]],[[11,[16,[11]],14,5,13],51],[[50,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],[[51,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],[[50,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],[[51,13,14,[16,[15]],[16,[17]],[16,[18]],[16,[19]]]],[[],2],[[],2],[[],2],[[],2],[[]],[[]],[[]],[[]],[[2,2,14,5,13],[[12,[51]]]],[[2,2,14,5,13],[[12,[51]]]],[[2,2,14,5,13],[[12,[50]]]],[[2,2,52],[[12,[48]]]],[[]],[[]],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],21],[[],21],[[],21],[[],21],[[]],[[]],[[]],[[]]],"c":[],"p":[[3,"Config"],[15,"usize"],[8,"Deserializer"],[4,"Result"],[15,"bool"],[3,"BertLinear"],[3,"Formatter"],[6,"Result"],[3,"LayerNorm"],[3,"BertModel"],[3,"Tensor"],[6,"Result"],[3,"LoraConfig"],[6,"VarBuilder"],[3,"LoraLinearConfig"],[4,"Option"],[3,"LoraConv1dConfig"],[3,"LoraConv2dConfig"],[3,"LoraEmbeddingConfig"],[15,"f64"],[3,"TypeId"],[3,"DinoVisionTransformer"],[3,"Falcon"],[3,"Config"],[3,"LlamaLinear"],[3,"Cache"],[3,"Config"],[3,"LlamaConfig"],[3,"Llama"],[4,"DType"],[4,"Device"],[3,"Shape"],[3,"Config"],[3,"Mistral"],[3,"Config"],[3,"Model"],[3,"T5EncoderModel"],[3,"T5ForConditionalGeneration"],[3,"Config"],[3,"Path"],[8,"AsRef"],[15,"slice"],[8,"SimpleBackend"],[3,"Box"],[3,"VarBuilderArgs"],[4,"Error"],[3,"TracedLoraConv2d"],[3,"QMatMul"],[3,"Conv2dConfig"],[3,"TracedLoraEmbedding"],[3,"TracedLoraLinear"],[3,"VarBuilder"]]}\
}');
if (typeof window !== 'undefined' && window.initSearch) {window.initSearch(searchIndex)};
if (typeof exports !== 'undefined') {exports.searchIndex = searchIndex};

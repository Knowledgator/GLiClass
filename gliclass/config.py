from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING

from .utils import is_module_available

IS_TURBOT5 = is_module_available('turbot5')

if IS_TURBOT5:
    from turbot5.model.config import T5Config
else:
    from transformers import T5Config


logger = logging.get_logger(__name__)


class GLiClassModelConfig(PretrainedConfig):
    model_type = "GLiClass"
    is_composition = True

    def __init__(
        self,
        encoder_config = None,
        encoder_model=None,
        label_model_config=None,
        label_model_name=None,
        class_token_index = -1,
        text_token_index = -1,
        example_token_index = -1,
        ignore_index=-100,
        hidden_size=None,
        projector_hidden_act="gelu",
        vocab_size=None,
        problem_type='single_label_classification',
        max_num_classes=25,
        use_lstm=False,
        initializer_range=0.03,
        scorer_type='simple',
        pooling_strategy='first',
        class_token_pooling="first",
        focal_loss_alpha=0.5,
        focal_loss_gamma=2,
        focal_loss_reduction=None,
        logit_scale_init_value=2.6592,
        normalize_features=False,
        extract_text_features=False,
        contrastive_loss_coef=0,
        architecture_type = 'uni-encoder',
        prompt_first = False,
        squeeze_layers = False,
        layer_wise = False,
        encoder_layer_id = -1,
        embed_class_token = True,
        dropout = 0.1,
        use_segment_embeddings = False,
        **kwargs,
    ):
        if isinstance(encoder_config, dict):
            encoder_config["model_type"] = (encoder_config["model_type"] 
                                                if "model_type" in encoder_config 
                                                else "deberta-v2")
            if encoder_config['model_type'] == 't5':
                encoder_config = T5Config(**encoder_config)
            else:
                encoder_config = CONFIG_MAPPING[encoder_config["model_type"]](**encoder_config)
        elif encoder_config is None:
            encoder_config = CONFIG_MAPPING["deberta-v2"]()

        self.encoder_config = encoder_config
        self.encoder_model_name = encoder_model

        if label_model_name is not None:
            if isinstance(label_model_config, dict):
                label_model_config["model_type"] = (label_model_config["model_type"] 
                                                    if "model_type" in label_model_config 
                                                    else "deberta-v2")
                label_model_config = CONFIG_MAPPING[label_model_config["model_type"]](**label_model_config)
            elif label_model_config is None:
                label_model_config = CONFIG_MAPPING["deberta-v2"]()

            self.label_model_config = label_model_config
        else:
            self.label_model_config = None
        self.label_model_name = label_model_name

        if hidden_size is None:
            self.hidden_size = self.encoder_config.hidden_size
        else:
            self.hidden_size = hidden_size

        if vocab_size is None:
            self.vocab_size = self.encoder_config.vocab_size
        else:
            self.vocab_size = vocab_size
        
        if class_token_index == -1:
            self.class_token_index = self.vocab_size
        else:
            self.class_token_index = class_token_index
        
        if text_token_index == -1:
            self.text_token_index = self.vocab_size+1
        else:
            self.text_token_index = text_token_index

        if example_token_index == -1:
            self.example_token_index = self.vocab_size+2
        else:
            self.example_token_index = example_token_index

        self.ignore_index = ignore_index
        self.projector_hidden_act = projector_hidden_act
        self.problem_type = problem_type
        self.max_num_classes = max_num_classes
        self.initializer_range=initializer_range
        self.scorer_type = scorer_type
        self.pooling_strategy=pooling_strategy
        self.class_token_pooling=class_token_pooling
        self.use_lstm = use_lstm
        self.focal_loss_alpha=focal_loss_alpha
        self.focal_loss_gamma=focal_loss_gamma
        self.focal_loss_reduction = focal_loss_reduction
        self.contrastive_loss_coef=contrastive_loss_coef
        self.logit_scale_init_value = logit_scale_init_value
        self.normalize_features=normalize_features
        self.extract_text_features = extract_text_features
        self.architecture_type = architecture_type
        self.prompt_first = prompt_first
        self.squeeze_layers = squeeze_layers
        self.layer_wise = layer_wise
        self.encoder_layer_id = encoder_layer_id
        self.embed_class_token = embed_class_token
        self.pad_token_id = self.encoder_config.pad_token_id
        self.dropout = dropout
        self.use_segment_embeddings = use_segment_embeddings
        super().__init__(**kwargs)


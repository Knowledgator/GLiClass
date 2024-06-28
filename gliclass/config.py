from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING
logger = logging.get_logger(__name__)


class GLiClassModelConfig(PretrainedConfig):
    model_type = "GLiClass"
    is_composition = True

    def __init__(
        self,
        encoder_config = None,
        encoder_model=None,
        class_token_index = -1,
        text_token_index = -1,
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
        focal_loss_alpha=0.5,
        focal_loss_gamma=2,
        logit_scale_init_value=2.6592,
        normalize_features=False,
        extract_text_features=False,
        contrastive_loss_coef=0,
        architecture_type = 'uni-encoder',
        **kwargs,
    ):
        if isinstance(encoder_config, dict):
            encoder_config["model_type"] = (encoder_config["model_type"] 
                                                if "model_type" in encoder_config 
                                                else "deberta-v2")
            encoder_config = CONFIG_MAPPING[encoder_config["model_type"]](**encoder_config)
        elif encoder_config is None:
            encoder_config = CONFIG_MAPPING["deberta-v2"]()

        self.encoder_config = encoder_config
        self.encoder_model_name = encoder_model

        if hidden_size is None:
            self.hidden_size = self.encoder_config.hidden_size
        else:
            self.hidden_size = hidden_size

        if vocab_size is None:
            self.vocab_size = self.encoder_config.vocab_size
        else:
            self.vocab_size = vocab_size
        
        if class_token_index == -1:
            self.class_token_index = vocab_size
        else:
            self.class_token_index = class_token_index
        
        if text_token_index == -1:
            self.text_token_index = vocab_size+1
        else:
            self.text_token_index = text_token_index

        self.ignore_index = ignore_index
        self.projector_hidden_act = projector_hidden_act
        self.problem_type = problem_type
        self.max_num_classes = max_num_classes
        self.initializer_range=initializer_range
        self.scorer_type = scorer_type
        self.pooling_strategy=pooling_strategy
        self.use_lstm = use_lstm
        self.focal_loss_alpha=focal_loss_alpha
        self.focal_loss_gamma=focal_loss_gamma
        self.contrastive_loss_coef=contrastive_loss_coef
        self.logit_scale_init_value = logit_scale_init_value
        self.normalize_features=normalize_features
        self.extract_text_features = extract_text_features
        self.architecture_type = architecture_type
        super().__init__(**kwargs)


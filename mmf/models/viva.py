import torch
 
# registry is need to register our new model so as to be MMF discoverable
from mmf.common.registry import registry
# All model using MMF need to inherit BaseModel
from mmf.models.base_model import BaseModel
# ProjectionEmbedding will act as proxy encoder for FastText Sentence Vector
from mmf.modules.embeddings import ProjectionEmbedding
# Builder methods for image encoder and classifier
from mmf.utils.build import build_classifier_layer, build_encoder


@registry.register_model("viva_model")
class Viva_model(BaseModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/viva/defaults.yaml"

    def build(self):
        self.num_labels = self.config.classifier.num_labels
        self.video_module = build_encoder(self.config.video_encoder)  
        self.audio_module = build_encoder(self.config.audio_encoder)  
        self.text_module = build_encoder(self.config.text_encoder)
        self.tabular_transform = torch.nn.Linear(self.config.tabular_transforms.in_dim, self.config.tabular_transforms.out_dim)
        self.tabular_batchnorm = torch.nn.BatchNorm1d(self.config.tabular_transforms.out_dim)
        self.classifier = build_classifier_layer(self.config.classifier)
        self.dropout = torch.nn.Dropout(self.config.dropout)
        self.interaction_layer = torch.nn.Sequential(
                                torch.nn.Linear(self.config.interaction_layer.input_dim, self.config.interaction_layer.hidden_dim),
                                torch.nn.ReLU(),
                                torch.nn.Linear(self.config.interaction_layer.hidden_dim, self.config.interaction_layer.output_dim)
                            )

    def forward(self, sample_list):
        video = sample_list["video"]  
        video_features = self.video_module(video)
        #TO DO: pooling
        video_features = video_features[0][:, 0]

        audio = sample_list["audio"]  
        audio_features = self.audio_module(audio).squeeze()

        text = sample_list["text"]  
        input_ids = torch.stack([torch.tensor(x) for x in text])
        text_features = self.text_module(input_ids=input_ids)

        tabular = sample_list["tabular"] 
        tabular_features = self.tabular_transform(tabular)
        tabular_features = self.tabular_batchnorm(tabular_features)

        combined = torch.cat([audio_features, video_features, text_features, tabular_features], dim=1)
        interacted = self.interaction_layer(combined)
        interacted = self.dropout(interacted)
        scores = self.classifier(interacted)

        return {"scores": scores}
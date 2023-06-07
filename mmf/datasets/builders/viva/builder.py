from mmf.common.registry import registry
from mmf.datasets.builders.viva.dataset import VivaDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("viva")
class VivaBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="viva", dataset_class=VivaDataset, *args, **kwargs
    ):
        super().__init__(dataset_name)
        self.dataset_class = VivaDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/viva/defaults.yaml"

"""Upload, validation and profiling."""
from __future__ import annotations

from imputeaudit import config
from imputeaudit.data import loader, profiler
from imputeaudit.data.loader import ValidatedDataset


class DatasetService:
    def __init__(self, repository=None):
        self.repository = repository
        self._cache: dict[str, ValidatedDataset] = {}

    def bundled_names(self) -> list[str]:
        return list(config.DATASETS)

    def load_bundled(self, name: str) -> ValidatedDataset:
        if name not in self._cache:
            self._cache[name] = loader.load_bundled(name)
            self._register(self._cache[name])
        return self._cache[name]

    def load_upload(self, raw: bytes, target: str, name: str) -> ValidatedDataset:
        frame = loader.read_csv_bytes(raw)
        dataset = loader.validate_dataset(frame, target, name, loader.file_hash(raw))
        dataset.meta.update({"title": name, "domain": "user upload", "split": "stratified"})
        self._register(dataset)
        return dataset

    def profile(self, dataset: ValidatedDataset) -> dict:
        return profiler.profile(dataset)

    def _register(self, dataset: ValidatedDataset) -> None:
        if self.repository is not None:
            self.repository.save_dataset(dataset.file_hash[:16], dataset.name, dataset.file_hash, len(dataset.X),
                                         dataset.X.shape[1], dataset.target,
                                         {str(k): int(v) for k, v in dataset.y.value_counts().items()})

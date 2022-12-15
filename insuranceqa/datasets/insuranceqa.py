import torch
from torch.utils.data import Dataset
from pyspark.storagelevel import StorageLevel
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer


class InsuranceDataset(Dataset):
    def __init__(
        self,
        spark,
        database_name="insuranceqa",
        split="train",
        input_col="question_en",
        label_col="topic_en",
        storage_level=StorageLevel.MEMORY_ONLY,
        tokenizer="distilbert-base-uncased",
        max_length=512,
    ):
        super().__init__()
        self.input_col = input_col
        self.label_col = label_col
        self.df = spark.sql(f"select * from {database_name}.{split}").toPandas()
        self.length = len(self.df)
        self.class_mappings = self._get_class_mappings()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

    def _get_class_mappings(self):
        labels = self.df.topic_en.unique()

        indexes = LabelEncoder().fit_transform(labels)
        class_mappings = dict(zip(labels, indexes))
        return class_mappings

    def _encode_label(self, label):
        self._get_class_mappings()
        label_class = self.class_mappings[label]
        encoded_label = torch.nn.functional.one_hot(torch.tensor([label_class]), num_classes=len(self.class_mappings))
        return encoded_label.type(torch.float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.df.loc[idx, self.input_col]
        inputs = self.tokenizer(
            question,
            None,
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        labels = self.df.loc[idx, self.label_col]
        labels = self._encode_label(labels)[0]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": labels,
        }

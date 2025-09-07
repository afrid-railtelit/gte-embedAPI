import torch
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from torch import Tensor
import onnx
import onnx.checker
from typing import Dict, List

class GteEmbedService:
    def __init__(self):
        self.model_name: str = "thenlper/gte-base"
        self.max_length: int = 300
        self.device: torch.device = torch.device("cpu")  # Use CPU for export to match trace
        self.tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model: BertModel = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

    def MeanPool(self, last_hidden: Tensor, mask: Tensor) -> Tensor:
        m = mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def Embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("texts empty")
        if len(texts) > 100:
            raise ValueError("texts length exceeds 100")

        tok: Dict[str, Tensor] = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

        inputs: Dict[str, Tensor] = {
            k: v.to(self.device, non_blocking=True) for k, v in tok.items()
        }

        with torch.inference_mode():
            out = self.model(**inputs)
            emb: Tensor = self.MeanPool(out.last_hidden_state, inputs["attention_mask"]).cpu()
        
        result: List[List[float]] = [emb[i].tolist() for i in range(emb.size(0))]
        return result

class GteModelWithPooling(torch.nn.Module):
    def __init__(self, service: GteEmbedService):
        super().__init__()
        self.model: BertModel = service.model
        self.mean_pool = service.MeanPool

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.model(input_ids, attention_mask)
        embeddings = self.mean_pool(outputs.last_hidden_state, attention_mask)
        return embeddings

def export_to_onnx() -> None:
    service = GteEmbedService()

    input_texts = ["This is a sample input for testing the gte-base model."]
    batch_dict = service.tokenizer(
        input_texts,
        max_length=service.max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = batch_dict["input_ids"].to(service.device)
    attention_mask = batch_dict["attention_mask"].to(service.device)

    model_with_pooling = GteModelWithPooling(service).to(service.device)

    output_path = "gte-base.onnx"
    try:
        torch.onnx.export(
            model_with_pooling,
            (input_ids, attention_mask),
            output_path,
            opset_version=14,  # Use opset 14 to support scaled_dot_product_attention
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "embeddings": {0: "batch_size"},
            },
        )
        print(f"Model successfully exported to {output_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        raise

    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
    except Exception as e:
        print(f"Error validating ONNX model: {e}")
        raise

if __name__ == "__main__":
    export_to_onnx()
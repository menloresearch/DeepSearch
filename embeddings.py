from typing import List, Union

import torch
import torch.nn.functional as F
from langchain.embeddings.base import Embeddings
from transformers import AutoModel, AutoTokenizer

# Set a default model here
DEFAULT_MODEL_NAME = "avsolatorio/NoInstruct-small-Embedding-v0"


class CustomHuggingFaceEmbeddings(Embeddings):
    """
    A custom embeddings class that wraps a Hugging Face model for generating embeddings.

    Supports two modes:
    - "sentence": uses the [CLS] token representation for sentence/document embeddings.
    - "query": uses mean pooling over tokens (weighted by the attention mask) for query embeddings.
    """

    def __init__(
        self, model_name: str = DEFAULT_MODEL_NAME, default_mode: str = "sentence"
    ):
        self.model_name = model_name
        # Set device to GPU if available, else CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.default_mode = default_mode  # "sentence" or "query"
        self.model.eval()  # Set model to evaluation mode

    def get_embedding(self, text: Union[str, List[str]], mode: str = None):
        if mode is None:
            mode = self.default_mode
        assert mode in (
            "query",
            "sentence",
        ), f"Unsupported mode: {mode}. Only 'query' and 'sentence' are supported."

        # Ensure we are working with a list of texts
        if isinstance(text, str):
            text = [text]

        # Tokenize the input texts
        inp = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # Move the input tensors to the same device as the model
        inp = {key: value.to(self.device) for key, value in inp.items()}

        # Forward pass (no gradients needed)
        with torch.no_grad():
            output = self.model(**inp)

        if mode == "query":
            # Mean pooling: weight by attention mask and average across tokens
            vectors = output.last_hidden_state * inp["attention_mask"].unsqueeze(2)
            vectors = vectors.sum(dim=1) / inp["attention_mask"].sum(dim=-1).view(-1, 1)
        else:
            # Sentence/document embedding: use the [CLS] token (first token) representation
            vectors = output.last_hidden_state[:, 0, :]
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for a list of documents (using sentence mode).
        """
        vectors = self.get_embedding(texts, mode="sentence")
        return vectors.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Compute an embedding for a single query.
        """
        vector = self.get_embedding(text, mode="query")
        return vector.cpu().numpy()[0].tolist()


# For quick testing
if __name__ == "__main__":
    embeddings = CustomHuggingFaceEmbeddings()

    # Example texts for document embeddings
    texts = [
        "Illustration of the REaLTabFormer model. The left block shows the non-relational tabular data model using GPT-2.",
        "Predicting human mobility holds significant practical value, with applications in disaster planning and epidemic simulation.",
        "As economies adopt digital technologies, policy makers are asking how to prepare the workforce for emerging labor demands.",
    ]
    doc_embeddings = embeddings.embed_documents(texts)
    print("Document embeddings:", doc_embeddings)

    # Example query embedding
    query_embedding = embeddings.embed_query("Which sentence talks about jobs?")
    print("Query embedding:", query_embedding)

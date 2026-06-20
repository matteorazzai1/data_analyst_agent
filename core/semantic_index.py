from typing import Any, Dict, List, Optional
import faiss
from sentence_transformers import SentenceTransformer


class SemanticIndex:
    """Indice vettoriale generico: embedding + FAISS (cosine similarity via
    normalizzazione L2 + IndexFlatIP). Ogni voce porta con sé un dict di
    metadata arbitrario, restituito tale e quale in fase di query.
    """

    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder
        self.index: Optional[faiss.Index] = None
        self.items: List[Dict[str, Any]] = []

    def build(self, texts: List[str], items: List[Dict[str, Any]]):
        assert len(texts) == len(items), "texts e items devono avere la stessa lunghezza"
        self.items = items
        if not texts:
            self.index = None
            return

        embeddings = self.encoder.encode(
            texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(embeddings)  # necessario per usare IndexFlatIP come cosine similarity

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(
        self,
        query: str,
        top_n: int = 5,
        filter_fn: Optional[callable] = None,
        over_fetch: int = 4,
    ) -> List[Dict[str, Any]]:
        """Cerca i top_n item più simili semanticamente alla query.

        filter_fn: funzione opzionale (item -> bool) per pre/post-filtrare
        i risultati per metadata (es. solo footnote di un certo indicator_code).
        Quando è presente un filtro, "sovra-peschiamo" più candidati prima di
        filtrare, perché FAISS IndexFlatIP non supporta filtri nativi.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        qv = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(qv)

        k = top_n * over_fetch if filter_fn else top_n
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(qv, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            item = dict(self.items[idx])
            if filter_fn and not filter_fn(item):
                continue
            item["score"] = float(score)
            results.append(item)
            if len(results) >= top_n:
                break
        return results

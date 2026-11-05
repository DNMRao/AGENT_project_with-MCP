"""Embed chunks with Sentence Transformers and search them with Qdrant."""

from dataclasses import asdict, dataclass

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_REVISION, MIN_SCORE, TOP_K, VECTOR_STORAGE_PATH
from src.ingestion import Chunk, build_chunks, load_documents

COLLECTION = "northstar_kb"

# Phrases that mark text as an instruction attack (see retrieved-content-safety.md).
# A keyword list only catches known wording; it is not a general injection defense.
UNSAFE_MARKERS = (
    "ignore all prior instructions",
    "ignore previous instructions",
    "reveal the system prompt",
    "api keys",
    "hidden tool outputs",
)


def is_unsafe(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in UNSAFE_MARKERS)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float  # cosine similarity, higher = more similar


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.size = len(chunks)
        self.model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cpu")
        self.client = QdrantClient(path=VECTOR_STORAGE_PATH) if VECTOR_STORAGE_PATH else QdrantClient(":memory:")
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)  # always rebuild from the current Markdown

        # 1. Create an empty collection: one vector per chunk, compared by cosine similarity.
        self.client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=self.model.get_embedding_dimension(),  # 384 for MiniLM
                distance=models.Distance.COSINE,
            ),
        )

        # 2. Embed all chunk texts in one batch (shape: n_chunks x 384).
        vectors = self.model.encode([c.text for c in chunks], normalize_embeddings=True)

        # 3. Store each vector with its chunk metadata as the payload.
        self.client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(id=i, vector=vec.tolist(), payload=asdict(chunk))
                for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )

    def search(self, question: str, k: int = TOP_K) -> list[SearchHit]:
        """Return the k chunks most similar to the question, best first."""
        qvec = self.model.encode(question, normalize_embeddings=True)
        result = self.client.query_points(
            COLLECTION, query=qvec.tolist(), limit=k, with_payload=True
        )
        return [SearchHit(chunk=Chunk(**p.payload), score=p.score) for p in result.points]

    def retrieve_evidence(
        self, question: str, k: int = TOP_K, min_score: float = MIN_SCORE
    ) -> tuple[list[SearchHit], list[str]]:
        """Up to k hits that each clear min_score and are safe, plus safety events.

        Unlike Module 5 (gate on the top-1 score only), every hit must clear the
        threshold here: a weakly related policy that slips into the top k would
        otherwise be cited, and the agent's citations must name only real support.
        An empty list means "no evidence" and the agent abstains.
        """
        hits: list[SearchHit] = []
        events: list[str] = []
        for hit in self.search(question, k=self.size):  # rank everything, then filter
            if len(hits) == k or hit.score < min_score:
                break  # hits are sorted, so nothing later can qualify
            if is_unsafe(hit.chunk.text):
                events.append(f"removed_unsafe_chunk:{hit.chunk.chunk_id}")
                continue
            hits.append(hit)
        return hits, events


if __name__ == "__main__":
    chunks = build_chunks(load_documents())
    retriever = Retriever(chunks)
    print(f"Indexed {len(chunks)} chunks with {EMBEDDING_MODEL}@{EMBEDDING_REVISION[:8]}")
    print(f"top_k={TOP_K}  min_score={MIN_SCORE}\n")

    for question in [
        "The customer was charged twice for one invoice.",  # should be answerable
        "What is the office Wi-Fi password?",  # should abstain
    ]:
        print(f"Q: {question}")
        for hit in retriever.search(question):
            print(f"  {hit.score:.3f}  {hit.chunk.chunk_id}")
        evidence, events = retriever.retrieve_evidence(question)
        decision = f"ANSWER from {[h.chunk.chunk_id for h in evidence]}" if evidence else "ABSTAIN"
        print(f"  -> {decision} {events or ''}\n")

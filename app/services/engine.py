import html
import hashlib
import json
import re
import time
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from app.core.index_client import VectorIndexClient
from app.ml.chunking import chunk_text
from app.ml.distill import distill_chunk
from app.ml.embeddings import Embedder
from app.services.source_registry import SourceRegistry
from app.services.telemetry import Telemetry
from app.services.blockchain import BlockchainService


def _strip_noise(text: str) -> str:
    text = html.unescape(text)
    # Remove fenced code blocks first.
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Remove inline code and markdown symbols.
    text = re.sub(r"`+", " ", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~>\[\]\(\)]", " ", text)
    # Normalize whitespace/newlines.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate_sentences(text: str, max_chars: int = 220, max_sentences: int = 2) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    total = 0
    for part in parts:
        if not part:
            continue
        candidate_len = total + len(part) + (1 if out else 0)
        if out and candidate_len > max_chars:
            break
        out.append(part)
        total = candidate_len
        if len(out) >= max_sentences:
            break
    if not out:
        return text[:max_chars].rstrip() + ("..." if len(text) > max_chars else "")
    result = " ".join(out).strip()
    if len(result) > max_chars:
        result = result[:max_chars].rstrip() + "..."
    return result


class MemoryEngine:
    def __init__(self) -> None:
        self.sources = SourceRegistry()
        self.embedder = Embedder()
        self.index = VectorIndexClient()
        self.crystals: Dict[str, dict] = {}
        self._vectors: List[np.ndarray] = []
        self._vector_ids: List[str] = []
        self.telemetry = Telemetry()
        self.blockchain = BlockchainService()  # Add blockchain service
        self.crystal_registry: Dict[str, dict] = {}
        self.query_usage: List[dict] = []
        self.wallets: Dict[str, dict] = {}
        self.transfers: List[dict] = []
        self._registry_dir = Path("data/registry")
        self._crystal_registry_path = self._registry_dir / "crystal_registry.json"
        self._query_usage_path = self._registry_dir / "query_usage.json"
        self._wallets_path = self._registry_dir / "wallets.json"
        self._transfers_path = self._registry_dir / "crystal_transfers.json"
        self._query_reward_pool = 0.01
        self._load_registry_store()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _hash_vector(self, vector: np.ndarray) -> str:
        arr = np.asarray(vector, dtype=np.float32)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def _compute_crystal_proof_hash(
        self,
        crystal_id: str,
        content_hash: str,
        embedding_hash: str,
        created_at: str,
    ) -> str:
        payload = {
            "crystal_id": crystal_id,
            "content_hash": content_hash,
            "embedding_hash": embedding_hash,
            "created_at": created_at,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _save_registry_store(self) -> None:
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        self._crystal_registry_path.write_text(
            json.dumps(self.crystal_registry, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        self._query_usage_path.write_text(
            json.dumps(self.query_usage, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        self._wallets_path.write_text(
            json.dumps(self.wallets, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        self._transfers_path.write_text(
            json.dumps(self.transfers, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _load_registry_store(self) -> None:
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        if self._crystal_registry_path.exists():
            try:
                self.crystal_registry = json.loads(self._crystal_registry_path.read_text(encoding="utf-8"))
            except Exception:
                self.crystal_registry = {}
        if self._query_usage_path.exists():
            try:
                raw = json.loads(self._query_usage_path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    self.query_usage = [self._normalize_query_usage_record(item) for item in raw]
                else:
                    self.query_usage = []
            except Exception:
                self.query_usage = []
        if self._wallets_path.exists():
            try:
                raw_wallets = json.loads(self._wallets_path.read_text(encoding="utf-8"))
                self.wallets = raw_wallets if isinstance(raw_wallets, dict) else {}
            except Exception:
                self.wallets = {}
        if self._transfers_path.exists():
            try:
                raw_transfers = json.loads(self._transfers_path.read_text(encoding="utf-8"))
                self.transfers = raw_transfers if isinstance(raw_transfers, list) else []
            except Exception:
                self.transfers = []
        self._normalize_wallets_and_registry()

    def _to_float(self, value: object, default: float = 0.0) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except Exception:
            return default

    def _normalize_crystals_used_entry(self, item: object, rank_fallback: int) -> dict:
        if isinstance(item, str):
            return {
                "crystal_id": item,
                "similarity": 0.0,
                "rank": rank_fallback,
                "contribution_score": 0.0,
                "reward_delta": 0.0,
            }
        if isinstance(item, dict):
            return {
                "crystal_id": str(item.get("crystal_id", "")),
                "similarity": self._to_float(item.get("similarity", 0.0)),
                "rank": int(item.get("rank", rank_fallback)),
                "contribution_score": self._to_float(item.get("contribution_score", 0.0)),
                "reward_delta": self._to_float(item.get("reward_delta", 0.0)),
            }
        return {
            "crystal_id": "",
            "similarity": 0.0,
            "rank": rank_fallback,
            "contribution_score": 0.0,
            "reward_delta": 0.0,
        }

    def _normalize_query_usage_record(self, record: object) -> dict:
        if not isinstance(record, dict):
            return {
                "query_id": f"query_{uuid.uuid4().hex[:12]}",
                "query_text": "",
                "created_at": self._now_iso(),
                "latency_ms": 0.0,
                "reward_pool": self._query_reward_pool,
                "crystals_used": [],
            }
        raw_used = record.get("crystals_used", [])
        used: list[dict] = []
        if isinstance(raw_used, list):
            used = [self._normalize_crystals_used_entry(item, idx + 1) for idx, item in enumerate(raw_used)]
        return {
            "query_id": str(record.get("query_id", f"query_{uuid.uuid4().hex[:12]}")),
            "query_text": str(record.get("query_text", "")),
            "created_at": str(record.get("created_at", self._now_iso())),
            "latency_ms": self._to_float(record.get("latency_ms", 0.0)),
            "reward_pool": self._to_float(record.get("reward_pool", self._query_reward_pool), self._query_reward_pool),
            "crystals_used": used,
        }

    def _normalize_wallet_record(self, wallet: object, owner_id_fallback: str) -> dict:
        if not isinstance(wallet, dict):
            return {
                "wallet_id": f"wallet_{owner_id_fallback}",
                "owner_id": owner_id_fallback,
                "created_at": self._now_iso(),
            }
        owner_id = str(wallet.get("owner_id", owner_id_fallback))
        return {
            "wallet_id": str(wallet.get("wallet_id", f"wallet_{owner_id}")),
            "owner_id": owner_id,
            "created_at": str(wallet.get("created_at", self._now_iso())),
        }

    def _normalize_transfer_record(self, record: object) -> dict:
        if not isinstance(record, dict):
            return {
                "transfer_id": f"transfer_{uuid.uuid4().hex[:12]}",
                "crystal_id": "",
                "from_owner_id": "",
                "to_owner_id": "",
                "transferred_at": self._now_iso(),
                "reason": "",
            }
        return {
            "transfer_id": str(record.get("transfer_id", f"transfer_{uuid.uuid4().hex[:12]}")),
            "crystal_id": str(record.get("crystal_id", "")),
            "from_owner_id": str(record.get("from_owner_id", "")),
            "to_owner_id": str(record.get("to_owner_id", "")),
            "transferred_at": str(record.get("transferred_at", self._now_iso())),
            "reason": str(record.get("reason", "")),
        }

    def _normalize_wallets_and_registry(self) -> None:
        normalized_registry: Dict[str, dict] = {}
        for crystal_id, item in self.crystal_registry.items():
            if not isinstance(item, dict):
                continue
            crystal_id_str = str(item.get("crystal_id", crystal_id))
            created_at = str(item.get("created_at", self._now_iso()))
            content_hash = str(item.get("content_hash", ""))
            embedding_hash = str(item.get("embedding_hash", ""))
            proof_hash = str(item.get("crystal_proof_hash", ""))
            if not proof_hash:
                proof_hash = self._compute_crystal_proof_hash(
                    crystal_id=crystal_id_str,
                    content_hash=content_hash,
                    embedding_hash=embedding_hash,
                    created_at=created_at,
                )
            owner_id = str(item.get("owner_id") or item.get("creator_id") or "local_user")
            normalized = dict(item)
            normalized["crystal_id"] = crystal_id_str
            normalized["created_at"] = created_at
            normalized["content_hash"] = content_hash
            normalized["embedding_hash"] = embedding_hash
            normalized["crystal_proof_hash"] = proof_hash
            normalized["owner_id"] = owner_id
            normalized_registry[crystal_id_str] = normalized
            self.get_or_create_wallet(owner_id)
        self.crystal_registry = normalized_registry

        normalized_wallets: Dict[str, dict] = {}
        for owner_id, wallet in self.wallets.items():
            rec = self._normalize_wallet_record(wallet, owner_id_fallback=str(owner_id))
            normalized_wallets[rec["owner_id"]] = rec
        self.wallets = normalized_wallets

        self.transfers = [self._normalize_transfer_record(item) for item in self.transfers]

    def _compute_contribution_scores(self, similarities: list[float]) -> list[float]:
        if not similarities:
            return []
        clamped = [max(float(s), 0.0) for s in similarities]
        total = float(sum(clamped))
        if total <= 1e-8:
            equal = 1.0 / float(len(clamped))
            return [equal for _ in clamped]
        return [s / total for s in clamped]

    def _upsert_registry_entry(
        self,
        crystal_id: str,
        source_url: str,
        creator_id: str,
        content: str,
        vector: np.ndarray,
    ) -> None:
        existing = self.crystal_registry.get(crystal_id, {})
        owner_id = str(existing.get("owner_id", creator_id))
        created_at = str(existing.get("created_at", self._now_iso()))
        content_hash = self._hash_text(content)
        embedding_hash = self._hash_vector(vector)
        crystal_proof_hash = self._compute_crystal_proof_hash(
            crystal_id=crystal_id,
            content_hash=content_hash,
            embedding_hash=embedding_hash,
            created_at=created_at,
        )
        # Submit to blockchain asynchronously (if this is a new crystal)
        is_new_crystal = crystal_id not in self.crystal_registry
        
        self.crystal_registry[crystal_id] = {
            "crystal_id": crystal_id,
            "creator_id": existing.get("creator_id", creator_id),
            "owner_id": owner_id,
            "source_url": existing.get("source_url", source_url),
            "content_hash": content_hash,
            "embedding_hash": embedding_hash,
            "crystal_proof_hash": crystal_proof_hash,
            "created_at": created_at,
            "usage_count": int(existing.get("usage_count", 0)),
            "reward_balance": float(existing.get("reward_balance", 0.0)),
            "contribution_total": float(existing.get("contribution_total", 0.0)),
            "last_contribution": float(existing.get("last_contribution", 0.0)),
            "last_reward_delta": float(existing.get("last_reward_delta", 0.0)),
            "blockchain_tx": existing.get("blockchain_tx", ""),  # Add blockchain transaction hash
            "blockchain_verified": existing.get("blockchain_verified", False),  # Add verification status
        }
        
        # Submit to blockchain if this is a new crystal
        if is_new_crystal:
            self._submit_to_blockchain(crystal_id, crystal_proof_hash, creator_id)
        
        self.get_or_create_wallet(owner_id)

    def _sync_registry_with_crystals(self) -> None:
        vector_map = {cid: vec for cid, vec in zip(self._vector_ids, self._vectors)}
        for crystal_id, crystal in self.crystals.items():
            vec = vector_map.get(crystal_id)
            if vec is None:
                continue
            self._upsert_registry_entry(
                crystal_id=crystal_id,
                source_url=str(crystal.get("source_ref", "unknown")),
                creator_id=str(crystal.get("metadata", {}).get("creator_id", "local_user")),
                content=str(crystal.get("fact_summary", "")),
                vector=vec,
            )

    def get_or_create_wallet(self, owner_id: str) -> dict:
        owner = str(owner_id or "local_user")
        wallet = self.wallets.get(owner)
        if wallet is None:
            wallet = {
                "wallet_id": f"wallet_{owner}",
                "owner_id": owner,
                "created_at": self._now_iso(),
            }
            self.wallets[owner] = wallet
        return wallet

    def list_wallet_crystals(self, owner_id: str, limit: int = 200) -> list[dict]:
        owner = str(owner_id)
        rows = [item for item in self.crystal_registry.values() if str(item.get("owner_id", "")) == owner]
        rows.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return rows[: max(limit, 0)]

    def compute_wallet_balance(self, owner_id: str) -> float:
        total = 0.0
        for item in self.crystal_registry.values():
            if str(item.get("owner_id", "")) != str(owner_id):
                continue
            total += float(item.get("reward_balance", 0.0))
        return float(total)

    def get_wallet_snapshot(self, owner_id: str) -> dict:
        wallet = self.get_or_create_wallet(owner_id)
        crystals = self.list_wallet_crystals(owner_id, limit=100000)
        return {
            "wallet_id": wallet["wallet_id"],
            "owner_id": wallet["owner_id"],
            "created_at": wallet["created_at"],
            "crystal_ids": [str(item.get("crystal_id", "")) for item in crystals if item.get("crystal_id")],
            "balance": self.compute_wallet_balance(owner_id),
            "crystal_count": len(crystals),
        }

    def list_wallet_transfers(self, owner_id: str, limit: int = 200) -> list[dict]:
        owner = str(owner_id)
        rows = [
            item
            for item in self.transfers
            if str(item.get("from_owner_id", "")) == owner or str(item.get("to_owner_id", "")) == owner
        ]
        rows.sort(key=lambda item: str(item.get("transferred_at", "")), reverse=True)
        return rows[: max(limit, 0)]

    def transfer_crystal(
        self,
        crystal_id: str,
        new_owner_id: str,
        actor_id: str | None = None,
        reason: str | None = None,
    ) -> dict:
        target = self.crystal_registry.get(crystal_id)
        if target is None:
            raise ValueError(f"Crystal not found: {crystal_id}")
        old_owner_id = str(target.get("owner_id") or target.get("creator_id") or "local_user")
        new_owner = str(new_owner_id or "").strip()
        if not new_owner:
            raise ValueError("new_owner_id is required")
        if new_owner == old_owner_id:
            raise ValueError("new_owner_id must be different from current owner")
        if actor_id is not None and str(actor_id).strip() and str(actor_id) != old_owner_id:
            raise ValueError("actor_id must match current owner")

        self.get_or_create_wallet(old_owner_id)
        self.get_or_create_wallet(new_owner)
        target["owner_id"] = new_owner
        transfer = {
            "transfer_id": f"transfer_{uuid.uuid4().hex[:12]}",
            "crystal_id": crystal_id,
            "from_owner_id": old_owner_id,
            "to_owner_id": new_owner,
            "transferred_at": self._now_iso(),
            "reason": str(reason or ""),
        }
        self.transfers.append(transfer)
        self.transfers = self.transfers[-5000:]
        self._save_registry_store()

        return {
            "transfer": transfer,
            "from_wallet": self.get_wallet_snapshot(old_owner_id),
            "to_wallet": self.get_wallet_snapshot(new_owner),
        }

    def _memory_stats(self) -> dict:
        crystal_bytes = sum(len(c["fact_summary"].encode("utf-8")) for c in self.crystals.values())
        compressed_bytes = sum(
            len(zlib.compress(c["fact_summary"].encode("utf-8"))) for c in self.crystals.values()
        )
        dim = int(self._vectors[0].shape[0]) if self._vectors else 0
        count = len(self._vectors)
        float_vector_bytes = count * dim * 4
        binary_vector_bytes = count * ((dim + 7) // 8)
        metadata_bytes = sum(
            len(str(c.get("crystal_id", "")))
            + len(str(c.get("source_type", "")))
            + len(str(c.get("source_ref", "")))
            for c in self.crystals.values()
        )
        total_footprint_bytes = crystal_bytes + float_vector_bytes + binary_vector_bytes + metadata_bytes
        return {
            "crystal_bytes": crystal_bytes,
            "compressed_bytes": compressed_bytes,
            "float_vector_bytes": float_vector_bytes,
            "binary_vector_bytes": binary_vector_bytes,
            "total_footprint_bytes": total_footprint_bytes,
        }

    def _exact_overlap(self, query_vector: np.ndarray, approx_ids: List[str], top_k: int) -> float:
        if not self._vectors or not approx_ids:
            return 0.0
        vecs = np.vstack(self._vectors)
        q = query_vector.astype(np.float32)
        q_norm = np.linalg.norm(q) + 1e-8
        v_norm = np.linalg.norm(vecs, axis=1) + 1e-8
        scores = (vecs @ q) / (v_norm * q_norm)
        idx = np.argsort(-scores)[:top_k]
        exact_ids = [self._vector_ids[i] for i in idx]
        return len(set(exact_ids).intersection(set(approx_ids))) / float(len(exact_ids) or 1)

    def get_metrics(self) -> dict:
        mem = self._memory_stats()
        return self.telemetry.snapshot(
            crystal_count=len(self.crystals),
            crystal_bytes=mem["crystal_bytes"],
            compressed_bytes=mem["compressed_bytes"],
            index_size=self.index.size(),
            float_vector_bytes=mem["float_vector_bytes"],
            binary_vector_bytes=mem["binary_vector_bytes"],
            total_footprint_bytes=mem["total_footprint_bytes"],
        )

    def set_index_mode(self, mode: str) -> dict:
        self.index.reconfigure(mode)
        if self._vectors and self._vector_ids:
            matrix = np.vstack(self._vectors).astype(np.float32)
            self.index.add(self._vector_ids, matrix)
        return {"mode": self.index.mode, "index_size": self.index.size()}

    def save_state(self, path: str = "data/state") -> dict:
        state_dir = Path(path)
        state_dir.mkdir(parents=True, exist_ok=True)

        crystals_path = state_dir / "crystals.json"
        vectors_path = state_dir / "vectors.npy"
        vector_ids_path = state_dir / "vector_ids.json"
        index_path = state_dir / "index_state.bin"

        crystals_path.write_text(json.dumps(self.crystals, ensure_ascii=True), encoding="utf-8")
        np.save(vectors_path, np.vstack(self._vectors).astype(np.float32) if self._vectors else np.zeros((0, 384), dtype=np.float32))
        vector_ids_path.write_text(json.dumps(self._vector_ids, ensure_ascii=True), encoding="utf-8")
        self.index.save_state(str(index_path))

        return {"saved": True, "path": str(state_dir), "items": len(self._vector_ids)}

    def load_state(self, path: str = "data/state") -> dict:
        state_dir = Path(path)
        crystals_path = state_dir / "crystals.json"
        vectors_path = state_dir / "vectors.npy"
        vector_ids_path = state_dir / "vector_ids.json"
        index_path = state_dir / "index_state.bin"

        if not (crystals_path.exists() and vectors_path.exists() and vector_ids_path.exists() and index_path.exists()):
            raise FileNotFoundError(f"State files not found under: {state_dir}")

        self.crystals = json.loads(crystals_path.read_text(encoding="utf-8"))
        vectors = np.load(vectors_path)
        self._vectors = [np.asarray(row, dtype=np.float32) for row in vectors]
        self._vector_ids = [str(x) for x in json.loads(vector_ids_path.read_text(encoding="utf-8"))]
        self.index.load_state(str(index_path))
        self._sync_registry_with_crystals()
        self._save_registry_store()

        return {"loaded": True, "path": str(state_dir), "items": len(self._vector_ids), "mode": self.index.mode}

    def ingest_text(
        self,
        text: str,
        source_type: str = "text",
        source_ref: str = "manual",
        creator_id: str = "local_user",
        source_url: str | None = None,
    ) -> dict:
        start = time.perf_counter()
        chunks = chunk_text(text)
        if not chunks:
            return {"ingested": 0, "message": "No usable text found."}

        distilled = [distill_chunk(chunk) for chunk in chunks]
        summaries = [item["fact_summary"] for item in distilled]
        vectors = self.embedder.encode(summaries)

        ids: List[str] = []
        for idx, summary in enumerate(summaries):
            crystal_id = str(uuid.uuid4())
            ids.append(crystal_id)
            self._vector_ids.append(crystal_id)
            self._vectors.append(vectors[idx])
            self.crystals[crystal_id] = {
                "crystal_id": crystal_id,
                "source_type": source_type,
                "source_ref": source_ref,
                "chunk_index": idx,
                "fact_summary": summary,
                "metadata": {"creator_id": creator_id},
            }
            self._upsert_registry_entry(
                crystal_id=crystal_id,
                source_url=source_url or source_ref,
                creator_id=creator_id,
                content=summary,
                vector=vectors[idx],
            )

        self.index.add(ids, vectors)
        self._save_registry_store()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.telemetry.record_ingest(
            chunks=len(ids),
            elapsed_ms=elapsed_ms,
            source_type=source_type,
            used_fallback=False,
        )
        return {"ingested": len(ids), "index_mode": self.index.mode, "elapsed_ms": elapsed_ms}

    def ingest_from_source(self, source_type: str, payload: dict) -> dict:
        if source_type == "text":
            text = payload.get("text", "")
            return self.ingest_text(
                text=text,
                source_type="text",
                source_ref=payload.get("source_ref", "manual"),
                creator_id=payload.get("creator_id", "local_user"),
                source_url=payload.get("source_ref", "manual"),
            )

        adapter = self.sources.get(source_type)
        enabled, message = adapter.status()
        if not enabled:
            self.telemetry.record_ingest(
                chunks=0,
                elapsed_ms=0.0,
                source_type=source_type,
                used_fallback=True,
            )
            return {
                "ingested": 0,
                "message": f"{source_type} unavailable: {message}",
                "fallback": "Use pdf/url/text.",
            }

        text = adapter.read(payload)
        source_ref = payload.get("source_ref") or payload.get("url") or payload.get("file_path") or source_type
        return self.ingest_text(
            text=text,
            source_type=source_type,
            source_ref=source_ref,
            creator_id=payload.get("creator_id", "local_user"),
            source_url=payload.get("url") or source_ref,
        )

    def query(self, query: str, top_k: int = 5, source_types: list[str] | None = None) -> dict:
        query_id = f"query_{uuid.uuid4().hex[:12]}"
        start = time.perf_counter()
        qv = self.embedder.encode([query])[0]
        hits, retrieval_stats = self.index.search_with_stats(qv, top_k=top_k)
        crystals = []
        for crystal_id, score in hits:
            crystal = self.crystals.get(crystal_id)
            if crystal is None:
                continue
            item = dict(crystal)
            item["score"] = score
            cleaned = _strip_noise(item.get("fact_summary", ""))
            item["clean_summary"] = cleaned
            item["preview_summary"] = _truncate_sentences(cleaned, max_chars=220, max_sentences=2)
            crystals.append(item)

        if source_types:
            crystals = [c for c in crystals if c.get("source_type") in source_types]

        if not crystals:
            answer = "No relevant memory crystals found yet. Ingest documents first."
        else:
            top = crystals[:5]
            summary_line = _truncate_sentences(top[0].get("clean_summary", ""), max_chars=260, max_sentences=2)
            key_points = [
                f"- [{c.get('source_type', 'unknown').upper()}] {c.get('preview_summary', '')}"
                for c in top
            ]
            seen_sources: set[str] = set()
            source_lines: list[str] = []
            for c in top:
                source_key = f"{c.get('source_type', 'unknown').upper()} | {c.get('source_ref', 'unknown')}"
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)
                source_lines.append(f"- {source_key}")
            answer = (
                "Summary:\n"
                f"{summary_line}\n\n"
                "Key points:\n"
                + "\n".join(key_points)
                + "\n\nSources:\n"
                + "\n".join(source_lines)
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        approx_ids = [c["crystal_id"] for c in crystals]
        overlap = self._exact_overlap(qv, approx_ids, top_k)
        score_mean = float(np.mean([c["score"] for c in crystals])) if crystals else 0.0
        citation_precision_proxy = 1.0 if crystals else 0.0
        self.telemetry.record_query(
            latency_ms=elapsed_ms,
            overlap=overlap,
            score_mean=score_mean,
            citation_precision_proxy=citation_precision_proxy,
            prefilter_candidates=float(retrieval_stats.get("prefilter_candidates", 0.0)),
            prefilter_ms=float(retrieval_stats.get("prefilter_ms", 0.0)),
            rerank_ms=float(retrieval_stats.get("rerank_ms", 0.0)),
            total_ms=float(retrieval_stats.get("total_ms", 0.0)),
        )
        scored_usage = [
            (
                str(c.get("crystal_id", "")),
                float(c.get("score", 0.0)),
                rank,
            )
            for rank, c in enumerate(crystals, start=1)
            if c.get("crystal_id")
        ]
        contributions = self._compute_contribution_scores([item[1] for item in scored_usage])
        crystals_used = []
        for (crystal_id, similarity, rank), contribution_score in zip(scored_usage, contributions):
            reward_delta = self._query_reward_pool * contribution_score
            crystals_used.append(
                {
                    "crystal_id": crystal_id,
                    "similarity": similarity,
                    "rank": rank,
                    "contribution_score": contribution_score,
                    "reward_delta": reward_delta,
                }
            )
            item = self.crystal_registry.get(crystal_id)
            if item is None:
                continue
            item["usage_count"] = int(item.get("usage_count", 0)) + 1
            item["reward_balance"] = float(item.get("reward_balance", 0.0)) + reward_delta
            item["contribution_total"] = float(item.get("contribution_total", 0.0)) + contribution_score
            item["last_contribution"] = contribution_score
            item["last_reward_delta"] = reward_delta
        self.query_usage.append(
            {
                "query_id": query_id,
                "query_text": query,
                "created_at": self._now_iso(),
                "latency_ms": elapsed_ms,
                "reward_pool": self._query_reward_pool,
                "crystals_used": crystals_used,
            }
        )
        self.query_usage = self.query_usage[-1000:]
        self._save_registry_store()
        return {
            "answer": answer,
            "crystals": crystals,
            "metrics": {
                "query_id": query_id,
                "query_latency_ms": elapsed_ms,
                "topk_overlap_vs_exact": overlap,
                "retrieval": retrieval_stats,
            },
        }

    def get_crystal_registry(self, crystal_id: str) -> dict | None:
        return self.crystal_registry.get(crystal_id)

    def list_crystal_registry(self, limit: int = 200) -> list[dict]:
        records = list(self.crystal_registry.values())
        records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return records[: max(limit, 0)]

    def get_query_usage(self, query_id: str) -> dict | None:
        for item in self.query_usage:
            if item.get("query_id") == query_id:
                return item
        return None

    def list_query_usage(self, limit: int = 200) -> list[dict]:
        records = list(self.query_usage)
        records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return records[: max(limit, 0)]

    def get_registry_leaderboard(self, limit: int = 50) -> list[dict]:
        rows = list(self.crystal_registry.values())
        rows.sort(
            key=lambda item: (
                int(item.get("usage_count", 0)),
                float(item.get("reward_balance", 0.0)),
            ),
            reverse=True,
        )
        return rows[: max(limit, 0)]
    
    # ===== BLOCKCHAIN INTEGRATION METHODS =====
    
    def _submit_to_blockchain(self, crystal_id: str, proof_hash: str, creator_id: str) -> None:
        """Submit crystal proof hash to blockchain asynchronously."""
        try:
            import threading
            import asyncio
            
            def submit():
                try:
                    if self.blockchain.is_available():
                        # Create new event loop for this thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            tx_hash = loop.run_until_complete(
                                self.blockchain.store_crystal_hash(crystal_id, proof_hash, creator_id)
                            )
                            if tx_hash:
                                # Update registry with blockchain transaction
                                registry_entry = self.crystal_registry.get(crystal_id)
                                if registry_entry:
                                    registry_entry["blockchain_tx"] = tx_hash
                                    registry_entry["blockchain_verified"] = True
                                    self._save_registry_store()
                                    print(f"✅ Crystal {crystal_id[:8]}... stored on blockchain: {tx_hash}")
                            else:
                                print(f"⚠️  Failed to store crystal {crystal_id[:8]}... on blockchain")
                        finally:
                            loop.close()
                    else:
                        print("⚠️  Blockchain service not available")
                except Exception as e:
                    print(f"⚠️  Blockchain submission error: {e}")
            
            # Submit in background thread to avoid blocking
            thread = threading.Thread(target=submit, daemon=True)
            thread.start()
            
        except Exception as e:
            print(f"⚠️  Failed to start blockchain submission: {e}")
    
    def verify_crystal_on_blockchain(self, crystal_id: str) -> dict:
        """Verify a crystal's authenticity using blockchain."""
        try:
            if not self.blockchain.is_available():
                return {"verified": False, "error": "Blockchain service not available"}
                
            result = self.blockchain.verify_crystal_hash(crystal_id)
            if result:
                return result
            else:
                return {"verified": False, "error": "Verification failed"}
                
        except Exception as e:
            return {"verified": False, "error": str(e)}
    
    def get_blockchain_account_info(self) -> dict:
        """Get blockchain account information."""
        try:
            return self.blockchain.get_account_info()
        except Exception as e:
            return {
                "address": None,
                "connected": False,
                "balance": "0",
                "network": "Unknown", 
                "error": str(e)
            }
    
    def get_blockchain_status(self) -> dict:
        """Get overall blockchain integration status."""
        account_info = self.get_blockchain_account_info()
        
        verified_crystals = sum(
            1 for crystal in self.crystal_registry.values() 
            if crystal.get("blockchain_verified", False)
        )
        
        pending_crystals = sum(
            1 for crystal in self.crystal_registry.values() 
            if crystal.get("blockchain_tx", "") and not crystal.get("blockchain_verified", False)
        )
        
        return {
            "available": self.blockchain.is_available(),
            "account": account_info,
            "crystals_verified": verified_crystals,
            "crystals_pending": pending_crystals,
            "total_crystals": len(self.crystal_registry)
        }

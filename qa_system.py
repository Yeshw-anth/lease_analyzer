import logging
import os
from typing import List, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from .pdf_parser import LeaseChunk
from .llm_provider import GeminiProvider, OllamaProvider, OpenAIProvider
from .models import LLMResponse
from collections import Counter
from . import deterministic_extractor as de

# --- Constants ---
BI_ENCODER_MODEL = 'BAAI/bge-base-en-v1.5'
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- Field Classification ---
FIELD_TYPES = {
    "Tenant": "deterministic",
    "Landlord": "deterministic",
    "Lease Date": "hybrid",
    "Lease Start Date": "hybrid",
    "Lease End Date": "hybrid",
    "Rent Amount": "deterministic",
    "Security Deposit": "deterministic",
    "Lease Term": "hybrid",
    "Renewal Term Length": "narrative",
    "Governing Law": "narrative",
    "Termination Clauses": "narrative"
    # All other fields will default to "narrative"
}

CRITICAL_FIELD_QUERY_VARIATIONS = {
    "Tenant": [
        "Who is the Tenant?",
        "Identify the Lessee.",
        "Who is the party leasing the property?"
    ],
    "Landlord": [
        "Who is the Landlord?",
        "Identify the Lessor.",
        "Who is the owner of the leased property?"
    ],
    "Lease Start Date": [
        "What is the Lease Commencement Date?",
        "What is the start date of the lease term?",
        "On what date does the lease term begin?"
    ],
    "Lease End Date": [
        "What is the Lease Expiration Date?",
        "When does the lease term end?",
        "On what date does the lease agreement terminate?"
    ],
    "Rent Amount": [
        "What is the monthly rent amount?",
        "How much is the base rent?",
        "What are the periodic rent payments?"
    ],
    "Renewal Options": [
        "Are there any options to renew the lease?",
        "Describe the lease renewal terms.",
        "What is the process for extending the lease term?"
    ],
    "Termination Clauses": [
        "Under what conditions can the lease be terminated?",
        "Describe the termination rights of the landlord and tenant.",
        "What are the provisions for early termination of the lease?"
    ],
    "Security Deposit": [
        "What is the amount of the security deposit?",
        "Is a security deposit required, and if so, how much?",
        "Describe the terms for the security deposit."
    ],
    "Special Provisions": [
        "Are there any special provisions or addenda in the lease?",
        "List any unique or non-standard clauses.",
        "Describe any special conditions or covenants in the agreement."
    ]
}

DETERMINISTIC_MAPPING = {
    "Tenant": lambda text: de.extract_entity(text, "Tenant"),
    "Landlord": lambda text: de.extract_entity(text, "Landlord"),
    "Rent Amount": de.extract_rent,
    "Lease Term": de.extract_lease_term,
    "Security Deposit": de.extract_security_deposit
}

class LeaseQuerySystem:
    def __init__(self, chunks: List[LeaseChunk], llm_provider_name: str = "gemini", api_key: Optional[str] = None):
        self.chunks = chunks
        self.text_by_chunk = {i: chunk.raw_text for i, chunk in enumerate(chunks)}
        self.full_text = "\n".join(self.text_by_chunk.values())
        self.embedder = SentenceTransformer(BI_ENCODER_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        self.index = self._build_index()
        self.llm_provider = self._get_llm_provider(llm_provider_name, api_key=api_key)
        self.extraction_stats = Counter()

    def _get_llm_provider(self, provider_name: str, api_key: Optional[str] = None):
        if provider_name == "gemini":
            if not api_key:
                api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API Key not provided. Please enter it in the sidebar or set the GEMINI_API_KEY environment variable.")
            return GeminiProvider(api_key=api_key)
        elif provider_name == "ollama":
            return OllamaProvider()
        elif provider_name == "openai":
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API Key not provided. Please enter it in the sidebar or set the OPENAI_API_KEY environment variable.")
            return OpenAIProvider(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

    def _build_index(self):
        logging.info("Building FAISS index...")
        embeddings = self.embedder.encode(list(self.text_by_chunk.values()), convert_to_tensor=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.cpu().numpy())
        logging.info("FAISS index built successfully.")
        return index

    def _construct_prompt(self, query: str, context: str) -> str:
        return f"Answer strictly based on the provided context. If the information is not explicitly stated, return 'Not Explicitly Stated'. Do not infer.\n\nContext: {context}\n\nQuery: {query}"

    def get_stats(self):
        return self.extraction_stats

    def run_queries(self, queries: List[str]) -> Dict[str, LLMResponse]:
        results = {}
        for query in queries:
            logging.info(f"Processing query: '{query}'")
            field_type = FIELD_TYPES.get(query, "narrative")
            
            response = None

            # Tier 1: Deterministic Extraction
            if field_type in ["deterministic", "hybrid"]:
                extractor = DETERMINISTIC_MAPPING.get(query)
                if extractor:
                    value, confidence = extractor(self.full_text)
                    if confidence == de.HIGH_CONFIDENCE:
                        response = LLMResponse(
                            answer=value,
                            source_chunk_text="Deterministic Extraction",
                            page_number=None,
                            section_number=None
                        )
                        self.extraction_stats['deterministic_success'] += 1

            # Tier 2: RAG Fallback
            if response is None:
                if field_type == "deterministic" or field_type == "hybrid":
                    if field_type == "deterministic":
                        logging.warning(f"Deterministic extraction failed for '{query}'. Falling back to RAG.")
                    self.extraction_stats['rag_fallback'] += 1
                else:  # narrative
                    self.extraction_stats['rag_direct'] += 1
                
                response = self._run_single_rag_query(query)

            # Tier 3: Multi-Query RAG for critical fields if initial attempts fail
            if response and ("not found" in response.answer.lower() or "not explicitly stated" in response.answer.lower()) and query in CRITICAL_FIELD_QUERY_VARIATIONS:
                logging.info(f"Initial RAG failed for critical field '{query}'. Trying query variations.")
                self.extraction_stats['multi_query_attempt'] += 1
                for alternative_query in CRITICAL_FIELD_QUERY_VARIATIONS[query]:
                    logging.info(f"Trying alternative query: '{alternative_query}'")
                    alternative_response = self._run_single_rag_query(alternative_query)
                    if alternative_response and "not found" not in alternative_response.answer.lower() and "not explicitly stated" not in alternative_response.answer.lower():
                        logging.info(f"Found answer with alternative query for '{query}'.")
                        response = alternative_response
                        self.extraction_stats['multi_query_success'] += 1
                        break 

            results[query] = response
        
        return results

    def _run_single_rag_query(self, query: str) -> LLMResponse:
        top_k = 3
        retrieved_chunks = self._retrieve_and_rerank(query, top_k=top_k)
        
        if not retrieved_chunks:
            self.extraction_stats['not_found'] += 1
            return LLMResponse(answer="Not Found in Document", source_chunk_text=None, page_number=None, section_number=None)

        top_chunk = retrieved_chunks[0]
        context = top_chunk.raw_text
        prompt = self._construct_prompt(query, context)
        
        try:
            raw_answer = self.llm_provider.generate(prompt)

            if "Not Explicitly Stated" in raw_answer:
                self.extraction_stats['not_found'] += 1
            else:
                self.extraction_stats['rag_success'] += 1

            return LLMResponse(
                answer=raw_answer,
                source_chunk_text=context,
                page_number=top_chunk.page_number,
                section_number=top_chunk.section_number
            )
        except Exception as e:
            logging.error(f"Error generating answer for query '{query}': {e}")
            self.extraction_stats['error'] += 1
            return LLMResponse(answer="Error during generation", source_chunk_text=context, page_number=top_chunk.page_number, section_number=top_chunk.section_number)

    def _retrieve_and_rerank(self, query: str, top_k: int) -> List[LeaseChunk]:
        query_embedding = self.embedder.encode(query, convert_to_numpy=True)
        distances, top_indices = self.index.search(np.array([query_embedding]), top_k)
        
        retrieved_chunks = [self.chunks[i] for i in top_indices[0]]
        
        cross_scores = self.cross_encoder.predict([(query, chunk.raw_text) for chunk in retrieved_chunks])
        
        reranked_indices = np.argsort(cross_scores)[::-1]
        reranked_chunks = [retrieved_chunks[i] for i in reranked_indices]
        
        return reranked_chunks
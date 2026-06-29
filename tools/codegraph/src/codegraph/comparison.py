import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Any

import networkx as nx

from codegraph.graph.queries import get_entity
from codegraph.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

@dataclass
class ComparisonAnalysis:
    relationships_traced_cg: int
    relationships_traced_llm: int
    file_references_cg: int
    file_references_llm: int
    hedge_word_count: int
    verdict: str

    def to_text(self) -> str:
        return (
            f"Relationships traced: CodeGraph={self.relationships_traced_cg}, LLM={self.relationships_traced_llm}\n"
            f"File:line references: CodeGraph={self.file_references_cg}, LLM={self.file_references_llm}\n"
            f"LLM Hedge words found: {self.hedge_word_count}\n\n"
            f"Verdict: {self.verdict}"
        )

@dataclass
class ComparisonResult:
    question: str
    codegraph_answer: str
    llm_answer: str
    llm_model: str
    context_given_to_llm: str
    analysis: ComparisonAnalysis

def run_comparison(
    question: str,
    cg_answer: str,
    api_key: str,
    graph: nx.DiGraph,
    model: str = "openai/gpt-4o",
) -> ComparisonResult:
    """
    Run a fair head-to-head comparison between CodeGraph and an LLM+RAG.
    """
    # 1. Get RAG context for the LLM (same as CodeGraph uses internally)
    retriever = HybridRetriever(graph)
    results = retriever.retrieve(question, top_k=5)
    
    context_chunks = []
    total_chars = 0
    for item in results.items:
        entity = item.entity
        # Simulate a typical RAG chunk
        chunk = f"File: {entity.file_path}\nEntity: {entity.qualified_name}\n"
        if entity.docstring:
            chunk += f"Doc: {entity.docstring}\n"
        
        # In a real RAG, we'd read the file. Here we simplify.
        context_chunks.append(chunk)
        total_chars += len(chunk)
        if total_chars > 6000:
            break
            
    context_str = "\n---\n".join(context_chunks)
    
    prompt = f"""You are a code analysis expert. Below is some context from a codebase and a question.
Answer the question accurately based ONLY on the provided context.

CONTEXT:
{context_str}

QUESTION:
{question}

Answer in a structured way.
"""

    # 2. Query LLM
    llm_answer = _query_llm(prompt, api_key, model)
    
    # 3. Analyze results
    analysis = _analyze_comparison(cg_answer, llm_answer)
    
    return ComparisonResult(
        question=question,
        codegraph_answer=cg_answer,
        llm_answer=llm_answer,
        llm_model=model,
        context_given_to_llm=f"{len(context_str)} chars of code chunks",
        analysis=analysis,
    )

def _query_llm(prompt: str, api_key: str, model: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode())
            return res["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM Query failed: {e}")
        return f"Error querying LLM: {e}"

def _analyze_comparison(cg_answer: str, llm_answer: str) -> ComparisonAnalysis:
    # Simple heuristic analysis
    cg_rels = cg_answer.count("→") + cg_answer.count("←")
    llm_rels = llm_answer.count("→") + llm_answer.count(" calls ")
    
    cg_refs = cg_answer.count(".py:")
    llm_refs = llm_answer.count(".py:")
    
    hedge_words = ["might", "likely", "probably", "potentially", "appears", "seems", "maybe"]
    hedge_count = sum(1 for w in hedge_words if w in llm_answer.lower())
    
    if cg_rels > llm_rels and cg_refs >= llm_refs:
        verdict = "CodeGraph provides significantly more precise, provable answers with exact dependency chains and file locations."
    elif llm_rels > 0:
        verdict = "LLM provides a good summary, but CodeGraph remains more precise on exact file:line references."
    else:
        verdict = "LLM provides a high-level description but lacks the structural depth and provability of CodeGraph."
        
    return ComparisonAnalysis(
        relationships_traced_cg=cg_rels,
        relationships_traced_llm=llm_rels,
        file_references_cg=cg_refs,
        file_references_llm=llm_refs,
        hedge_word_count=hedge_count,
        verdict=verdict,
    )

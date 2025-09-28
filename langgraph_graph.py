#for with confidence scoring and PDF input
import os
import requests
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#State Schema
class ResearchState(TypedDict, total=False):
    topic: str
    subquestions: List[str]
    findings: List[str]
    summary: str
    report: str
    citations: List[str]
    pdf_text: str
    attempted_replanning: bool
    confidence_scores: List[float]


#Tavilyy Search
def tavily_search(query: str) -> List[Dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "query": query,
        "include_answers": True,
        "include_raw_content": False,
        "include_images": False,
        "max_results": 5
    }
    response = requests.post(url, headers=headers, json=body)
    data = response.json()
    results = []
    for item in data.get("results", []):
        results.append({
            "question": query,
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", "")
        })
    return results

#Gathering Node
def gatherer_node(state: Dict) -> Dict:
    topic = state["topic"]
    findings = []
    citations = []

    search_results = tavily_search(topic)
    for item in search_results:
        findings.append(f"Q: {topic}\nA: {item['content']}")
        citations.append(item["url"])

    return {
        "topic": topic,
        "findings": findings,
        "citations": citations,
        "pdf_text": state.get("pdf_text", ""),
    }

#Scoring
def rate_confidence(text:str)-> float:
    prompt = f"Rate your confidence (0-100) in the factual accuracy of the following answer. Respond with just a number:\n\n{text}"
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        rating = response.choices[0].message.content.strip()
        return float(rating)
    except:
        return 50.0

#Synthesizer Node
def summarize_text(text: str, topic: str) -> str:
    prompt = f"Summarize this research note on '{topic}':\n\n{text[:4000]}"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def synthesizer_node(state: Dict) -> Dict:
    topic = state["topic"]
    partial_summaries = []
    confidence_scores = []

    pdf_context = state.get("pdf_text", "")
    if pdf_context:
        partial_summaries.append(f" Context from PDF:\n{pdf_context[:4000]}")

    for finding in state["findings"]:
        try:
            summary = summarize_text(finding, topic)
            partial_summaries.append(summary)
            confidence_scores.append(rate_confidence(summary))
        except Exception as e:
            partial_summaries.append(f"[Error summarizing one finding: {e}]")
            confidence_scores.append(50.0)

    combined = "\n\n".join(partial_summaries)[:8000]
    
    # Generate the detailed report
    report_prompt = f"Based on the following summaries and any provided PDF context, provide a very detailed, structured research report on '{topic}':\n\n{combined}"
    report_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": report_prompt}]
    )
    report = report_response.choices[0].message.content.strip()

    # Generate the executive summary
    summary_prompt = f"Create a concise, one-paragraph executive summary for the following report:\n\n{report}"
    summary_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": summary_prompt}]
    )
    summary = summary_response.choices[0].message.content.strip()

    return {
        "summary": summary,
        "report": report,
        "citations": state.get("citations", []),
        "confidence_scores": confidence_scores
    }

#Orchestrator
def run_research_agent(topic: str, pdf_text: str = "") -> Dict:
    builder = StateGraph(ResearchState)

    builder.add_node("Gatherer", gatherer_node)
    builder.add_node("Synthesizer", synthesizer_node)

    builder.set_entry_point("Gatherer")
    builder.add_edge("Gatherer", "Synthesizer")
    builder.add_edge("Synthesizer", END)

    graph = builder.compile()
    result = graph.invoke({"topic": topic, "pdf_text": pdf_text})
    return result

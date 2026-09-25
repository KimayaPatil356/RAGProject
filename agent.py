import argparse
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


# ==========================================
# KNOWLEDGE BASE
# ==========================================

SAMPLE_KB = [
    "Product: CloudSync Pro. Features: real-time sync across 5 devices, 1TB storage, offline mode, version history 30 days.",
    "Pricing: Basic $9/mo (100GB, 2 devices), Pro $19/mo (1TB, 5 devices), Business $49/mo (5TB, unlimited devices).",
    "Cancellation: Cancel anytime from Account > Subscription > Cancel. Refunds available within 14 days of charge.",
    "Password reset: Go to login page, click 'Forgot Password', enter email. Reset link expires in 1 hour.",
    "Sync issues: Check internet connection, ensure app is updated, try Sign Out and Sign In. If persists, contact support.",
    "Supported platforms: Windows 10+, macOS 12+, iOS 15+, Android 10+, Linux (Beta).",
    "Data security: AES-256 encryption at rest and in transit. SOC 2 Type II certified. Zero-knowledge architecture.",
    "File size limit: Individual files up to 10GB (Pro/Business), 2GB (Basic). No limit on total number of files.",
]


# ==========================================
# ESCALATION RULES
# ==========================================

ESCALATION_KEYWORDS = [
    "refund",
    "lawsuit",
    "furious",
    "fraud",
    "broken",
    "data loss",
    "cancel account",
    "charge",
    "billing error",
]


# ==========================================
# STATE
# ==========================================

class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    retrieved_context: str
    response: str
    escalate: bool

def retrieve_context(state: SupportState) -> SupportState:
    query = state["user_input"].lower()

    if "pricing" in query or "price" in query or "plans" in query:
        for text in getattr(retrieve_context, "kb_texts", SAMPLE_KB):
            if "basic: $9/month" in text.lower():
                return {"retrieved_context": text}

# ==========================================
# RETRIEVAL
# ==========================================

def retrieve_context(state: SupportState) -> SupportState:
    query = state["user_input"].lower()

    texts = getattr(
        retrieve_context,
        "kb_texts",
        SAMPLE_KB
    )

    topic_keywords = {
        "pricing": [
            "price", "pricing", "cost", "plans",
            "plan", "monthly", "subscription"
        ],
        "password": [
            "password", "reset", "forgot", "login"
        ],
        "cancel": [
            "cancel", "cancellation", "refund"
        ],
        "platform": [
            "platform", "windows", "macos", "ios",
            "android", "linux", "supported"
        ],
        "sync": [
            "sync", "synchronization", "offline",
            "internet", "not working"
        ],
        "security": [
            "security", "encryption", "soc",
            "zero-knowledge", "secure"
        ],
        "file": [
            "file size", "file limit", "upload",
            "gb", "files"
        ],
        "product": [
            "product", "cloudsync", "feature",
            "storage", "device"
        ],
    }

    matched_topics = []

    for topic, keywords in topic_keywords.items():
        if any(keyword in query for keyword in keywords):
            matched_topics.append(topic)

    scored_docs = []

    for text in texts:
       text_lower = text.lower()
       score = 0

       query_words = query.split()

       for word in query_words:
           if len(word) > 2 and word in text_lower:
              score += 1

    scored_docs.append((score, text))

    scored_docs.sort(
        key=lambda item: item[0],
        reverse=True
    )

    if scored_docs and scored_docs[0][0] > 0:
        context = scored_docs[0][1]
    else:
        context = (
            "No relevant information was found "
            "in the CloudSync Pro knowledge base."
        )

    return {
        "retrieved_context": context
    }
# ==========================================
# ESCALATION
# ==========================================

def check_escalation(state: SupportState) -> SupportState:

    text = state["user_input"].lower()

    needs_escalation = any(
        keyword in text
        for keyword in ESCALATION_KEYWORDS
    )

    return {
        "escalate": needs_escalation
    }


# ==========================================
# RESPONSE GENERATION
# ==========================================
def generate_response(state: SupportState) -> SupportState:

    if state.get("escalate"):

        response_text = (
            "ESCALATION REQUIRED\n\n"
            "This issue needs to be handled by a senior support specialist. "
            "Your case has been flagged for human review.\n"
            "Case ID: #"
            + str(hash(state["user_input"]) % 100000)
        )

    else:

        context = state["retrieved_context"]

        if not context.strip():

            response_text = (
                "I couldn't find enough information in the "
                "CloudSync Pro knowledge base to answer this question."
            )

        else:

           response_text = (
                 "Based on the CloudSync Pro knowledge base:\n\n"
                 + context
                 + "\n\nSources: CloudSync Pro Knowledge Base"
)

    return {
        "response": response_text,
        "messages": [
            AIMessage(content=response_text)
        ],
    }

# ==========================================
# GRAPH
# ==========================================

def build_graph():

    graph = StateGraph(SupportState)

    graph.add_node(
        "retrieve",
        retrieve_context
    )

    graph.add_node(
        "check_escalation",
        check_escalation
    )

    graph.add_node(
        "generate",
        generate_response
    )

    graph.set_entry_point("retrieve")

    graph.add_edge(
        "retrieve",
        "check_escalation"
    )

    graph.add_edge(
        "check_escalation",
        "generate"
    )

    graph.add_edge(
        "generate",
        END
    )

    return graph.compile()


# ==========================================
# LOAD CUSTOM KNOWLEDGE BASE
# ==========================================

def load_kb_texts(kb_dir: str | None) -> list[str]:

    if not kb_dir:
        return SAMPLE_KB

    root = Path(kb_dir)

    if not root.is_dir():
        raise ValueError(
            f"Knowledge base directory does not exist: {kb_dir}"
        )

    texts = []

    for path in sorted(root.rglob("*")):

        if path.is_file() and path.suffix.lower() in {
            ".txt",
            ".md"
        }:

            texts.append(
                path.read_text(
                    encoding="utf-8"
                )
            )

    if not texts:
        raise ValueError(
            f"No .txt or .md files found in knowledge base directory: {kb_dir}"
        )

    return texts


# ==========================================
# MAIN
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Customer Support Agent"
    )

    parser.add_argument(
        "--kb-dir",
        help="Directory containing .txt or .md support knowledge base files"
    )

    args = parser.parse_args()

    retrieve_context.kb_texts = load_kb_texts(
        args.kb_dir
    )

    if hasattr(
        retrieve_context,
        "retriever"
    ):
        delattr(
            retrieve_context,
            "retriever"
        )

    agent = build_graph()

    state = {
        "messages": [],
        "user_input": "",
        "retrieved_context": "",
        "response": "",
        "escalate": False,
    }

    print("\nCustomer Support Agent (CloudSync Pro)")
    print("Type 'quit' to exit\n")

    while True:

        user_input = input(
            "Customer: "
        ).strip()

        if user_input.lower() in (
            "quit",
            "exit",
            "q"
        ):
            break

        if not user_input:
            continue

        state["user_input"] = user_input

        state["messages"].append(
            HumanMessage(
                content=user_input
            )
        )

        state = agent.invoke(state)

        escalation_indicator = (
            " [ESCALATED]"
            if state.get("escalate")
            else ""
        )

        print(
            f"\nAgent{escalation_indicator}: "
            f"{state['response']}\n"
        )


if __name__ == "__main__":
    main()

import argparse
import json
import re
from typing import Any, Literal

from sql_lookup import SQLQueryError, query_database
from llm_client import (
    answer_or_verification_notice,
    get_groq_client,
    get_groq_model,
)
from sop_rag import answer_from_sop, search_sop


Route = Literal[
    "general", "policy", "orders", "both", "schema", "policy+schema"
]

_SCHEMA_QUESTION = re.compile(
    r"\b(schema|database structure|structure of "
    r"(?:(?:the|your|our|my) )?(?:data\s*base|db)|"
    r"(?:database|db) (?:tables?|columns?|fields?|types?|metadata)|"
    r"(?:tables?|columns?|fields?|data types?|constraints?) "
    r"(?:in|of|for) (?:the )?(?:database|data\s*base|db)|"
    r"(?:tables?|columns?|fields?|data types?|constraints?)"
    r".{0,40}(?:database|data\s*base|db)|"
    r"tables and columns)\b",
    flags=re.IGNORECASE,
)

_SCHEMA_REFUSAL = (
    "I can help with order information and SOP guidance, but I can't provide "
    "internal database structure or system configuration."
)


def _is_schema_question(question: str) -> bool:
    return _SCHEMA_QUESTION.search(question) is not None


def classify_question(question: str) -> Route:
    response = get_groq_client().chat.completions.create(
        model=get_groq_model(),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user's question for a supply-chain assistant. "
                    "Return exactly one label: general, policy, orders, both, "
                    "schema, or policy+schema.\n\n"
                    "Rules:\n"
                    "- general: ordinary general-knowledge questions that do "
                    "not require the project's SOP documents or live database. "
                    "Examples: 'What is supply chain?', 'What is logistics?', "
                    "'Explain inventory management in simple words'.\n"
                    "- policy: questions specifically about procedures, rules, "
                    "or guidance in the available SOP documents.\n"
                    "- orders: questions requiring live customer, order, or "
                    "order-item database facts.\n"
                    "- both: questions requiring BOTH SOP guidance and live "
                    "order data.\n"
                    "- schema: questions asking for internal database structure, "
                    "tables, columns, or types.\n"
                    "- policy+schema: questions requesting both SOP guidance "
                    "and internal database structure.\n\n"
                    "Do not choose orders or both unless live database facts "
                    "are actually needed. Do not choose policy merely because "
                    "a question mentions supply chains generally. If the "
                    "question can be answered with general knowledge, choose "
                    "general."
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    label = response.choices[0].message.content
    if not label:
        raise RuntimeError("The model returned an empty routing decision.")

    route = label.strip().lower()
    allowed_routes = {
        "general", "policy", "orders", "both", "schema", "policy+schema"
    }

    if route not in allowed_routes:
        # Fail safely: do not send an unclear question to PostgreSQL.
        return "general"

    return route  # type: ignore[return-value]


def _answer_general(question: str) -> str:
    response = get_groq_client().chat.completions.create(
        model=get_groq_model(),
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful supply-chain assistant. Answer general "
                    "knowledge questions clearly and accurately, using simple "
                    "language when appropriate. Do not query or claim to have "
                    "queried a database or SOP documents. If the user asks for "
                    "specific current order records, explain that those require "
                    "the order-data workflow."
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("The model returned an empty general answer.")

    return answer.strip()


def _answer_with_database(
    question: str, rows: list[dict[str, object]]
) -> str:
    response = get_groq_client().chat.completions.create(
        model=get_groq_model(),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's database question using only the supplied "
                    "query results. Do not invent missing values. If there are no "
                    "rows, say no matching records were found."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nDatabase query results:\n"
                    f"{json.dumps(rows, default=str)}"
                ),
            },
        ],
    )

    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("The model returned an empty database answer.")

    return answer_or_verification_notice(question, answer.strip(), rows)


def _answer_with_both(
    question: str,
    rows: list[dict[str, object]],
    sources: list[dict[str, Any]],
) -> str:
    sop_context = "\n\n".join(
        (
            f"Document: {source['document']}; page: {source['page']}\n"
            f"Text: {source['text']}"
        )
        for source in sources
    )

    response = get_groq_client().chat.completions.create(
        model=get_groq_model(),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the live database results and SOP passages "
                    "below. Clearly distinguish current database facts from policy "
                    "guidance, cite SOP document and page, and do not invent facts. "
                    "If either source lacks needed information, say so."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nLive database results:\n"
                    f"{json.dumps(rows, default=str)}\n\nSOP passages:\n"
                    f"{sop_context or '(No matching SOP passages found.)'}"
                ),
            },
        ],
    )

    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("The model returned an empty combined answer.")

    return answer_or_verification_notice(
        question,
        answer.strip(),
        {"database_rows": rows, "sop_passages": sources},
    )


def answer_question(question: str, top_k: int = 5) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    if _is_schema_question(question):
        return {"route": "restricted", "answer": _SCHEMA_REFUSAL}

    route = classify_question(question)

    if route in {"schema", "policy+schema"}:
        return {"route": "restricted", "answer": _SCHEMA_REFUSAL}

    # General questions use Groq directly, without FAISS or PostgreSQL.
    if route == "general":
        return {
            "route": "general",
            "answer": _answer_general(question),
        }

    # SOP questions use FAISS and the existing SOP answer workflow.
    if route == "policy":
        sources = search_sop(question, top_k=top_k)
        answer = answer_from_sop(question, sources)
        return {
            "route": "policy",
            "answer": answer,
            "sources": sources,
        }

    # Only order-related and combined questions reach PostgreSQL.
    try:
        sql, rows = query_database(question)
    except SQLQueryError:
        return {
            "route": route,
            "answer": (
                "I couldn't retrieve the requested order information. "
                "Please check the database connection and try again."
            ),
        }
    except (KeyError, RuntimeError) as exc:
        if "DATABASE_URL" in str(exc):
            return {
                "route": route,
                "answer": (
                    "Order information is unavailable because the database "
                    "connection is not configured. General and SOP questions "
                    "can still be answered without PostgreSQL."
                ),
            }
        raise

    if route == "orders":
        return {
            "route": "orders",
            "answer": _answer_with_database(question, rows),
            "sql": sql,
            "rows": rows,
        }

    sources = search_sop(question, top_k=top_k)

    return {
        "route": "both",
        "answer": _answer_with_both(question, rows, sources),
        "sql": sql,
        "rows": rows,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer questions using general knowledge, SOP documents, "
        "and order data."
    )
    parser.add_argument("question", help="Question to ask the assistant")
    args = parser.parse_args()

    result = answer_question(args.question)
    print(f"Route: {result['route']}")
    print(f"\n{result['answer']}")

    for source in result.get("sources", []):
        print(
            f"\nSource: {source['document']} "
            f"(page {source['page']}, chunk {source['chunk_number']})"
        )


if __name__ == "__main__":
    main()

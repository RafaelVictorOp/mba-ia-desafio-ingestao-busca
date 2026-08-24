from typing import Any
from sqlalchemy import create_engine, text
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_postgres import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

HISTÓRICO DA CONVERSA:
{historico}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


class SearchService:
    def __init__(self, envs: dict[str, Any], modelo: int):
        self._session_store: dict[str, InMemoryChatMessageHistory] = {}
        self._llm = self._create_llm(modelo)
        self._embeddings = self._create_embeddings(envs, modelo)
        self._store = PGVector(
            embeddings=self._embeddings,
            collection_name=envs["PG_VECTOR_COLLECTION_NAME"],
            connection=envs["DATABASE_URL"],
            use_jsonb=True,
        )
        self._prompt = PromptTemplate(
            input_variables=["contexto", "history", "pergunta"],
            template=PROMPT_TEMPLATE,
        )
        self._engine = create_engine(envs["DATABASE_URL"])

    def _create_llm(self, modelo: int) -> ChatOpenAI | ChatGoogleGenerativeAI:
        if modelo == 1:
            return ChatOpenAI(
                model="gpt-5.4-mini",
                temperature=0.9,
            )

        if modelo == 2:
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.9,
            )

        raise ValueError(f"Modelo inválido: {self.modelo}")

    def _create_embeddings(self, envs: dict[str, Any], modelo: int) -> OpenAIEmbeddings | GoogleGenerativeAIEmbeddings:
        if modelo == 1:
            valor = envs.get("OPENAI_EMBEDDING_MODEL")
            if valor is None: raise ValueError(f"OPENAI_EMBEDDING_MODEL está sem valor definido")

            return OpenAIEmbeddings(model=valor)

        if modelo == 2:
            valor = envs.get("GOOGLE_EMBEDDING_MODEL")

            if valor is None: raise ValueError(f"GOOGLE_EMBEDDING_MODEL está sem valor definido")

            return GoogleGenerativeAIEmbeddings(model=valor)

        raise ValueError(f"Modelo inválido: {self.modelo}")

        
    def search(self, question: str, session_id: str = "default") -> str:
        history = self._get_session_history(session_id)
        historico = "\n".join(
            f"{'Usuário' if message.type == 'human' else 'Assistente'}: "
            f"{message.content}"
            for message in history.messages
        )

        documentos_encontrados = self._read_context(question)
        context = "\n\n".join(document.page_content for document in documentos_encontrados)

        full_prompt = self._prompt.format(
            contexto=context,
            historico=historico,
            pergunta=question,
        )

        raw_answer = self._llm.invoke(full_prompt).content

        history.add_user_message(question)
        history.add_ai_message(raw_answer)

        return raw_answer

    def _get_session_history(
        self,
        session_id: str,
    ) -> InMemoryChatMessageHistory:
        if session_id not in self._session_store:
            self._session_store[session_id] = InMemoryChatMessageHistory()

        return self._session_store[session_id]

    def _read_context(self, question: str) -> list[Document]:
        return self._store.similarity_search(question, k=3)
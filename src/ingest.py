import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class PdfIngestionService:
    def __init__(self, envs: dict[str, any], modelo: int):
        self.envs = envs
        self.modelo = modelo

    def ingest(self) -> None:
        docs = self._load_pdf()
        splits = self._split_documents(docs)
        enriched = self._enrich_documents(splits)
        embeddings = (
            self._create_embeddings_open_ia()
            if self.modelo == 1
            else self._create_embeddings_google()
        )
        store = self._create_vector_store(embeddings)

        ids = [f"doc-{i}" for i in range(len(enriched))]

        store.add_documents(
            documents=enriched,
            ids=ids,
        )

        print(f"Arquivo {self.envs['PG_VECTOR_COLLECTION_NAME']} carregado com sucesso!")

    def _load_pdf(self) -> list[Document]:
        docs = PyPDFLoader(self.envs["PDF_PATH"]).load()

        if not docs:
            raise RuntimeError("PDF não encontrado.")

        return docs

    def _split_documents(
        self,
        docs: list[Document],
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            add_start_index=False,
        )

        return splitter.split_documents(docs)

    def _enrich_documents(
        self,
        docs: list[Document],
    ) -> list[Document]:
        return [
            Document(
                page_content=d.page_content,
                metadata={
                    k: v
                    for k, v in d.metadata.items()
                    if v not in ("", None)
                },
            )
            for d in docs
        ]

    def _create_embeddings_open_ia(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self.envs["OPENAI_EMBEDDING_MODEL"],
        )

    def _create_embeddings_google(self) -> GoogleGenerativeAIEmbeddings:
        return GoogleGenerativeAIEmbeddings(
            model=self.envs["GOOGLE_EMBEDDING_MODEL"],
        )

    def _create_vector_store(
        self,
        embeddings: OpenAIEmbeddings | GoogleGenerativeAIEmbeddings,
    ) -> PGVector:
        return PGVector(
            embeddings=embeddings,
            collection_name=self.envs["PG_VECTOR_COLLECTION_NAME"],
            connection=self.envs["DATABASE_URL"],
            use_jsonb=True,
        )
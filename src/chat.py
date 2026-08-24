from search import SearchService
import os
from typing import Any
from ingest import PdfIngestionService
from dotenv import load_dotenv

def main(search_service: SearchService):
    try:
        while True:
            question = input("Pergunta: ").strip()

            if not question:
                continue

            if question.lower() in ("sair", "exit", "quit"):
                break

            answer = search_service.search(question)
            print(f"\nAssistente: {answer}\n")
    except KeyboardInterrupt:
        pass

    print("\nEncerrando chat.")

def load_envs() -> dict[str, Any]:
    envs = {
        "OPENAI_API_KEY": None,
        "DATABASE_URL": None,
        "PG_VECTOR_COLLECTION_NAME": None,
        "PDF_PATH": None,
        "OPENAI_EMBEDDING_MODEL": None
    }

    for key in envs:
        value = os.getenv(key)

        if not value:
            raise RuntimeError(f"Environment variable {key} não foi definida")

        envs[key] = value

    return envs

if __name__ == "__main__":
    load_dotenv()
    envs = load_envs()

    modelo = None
    while True:
        try:
            modelo = int(input(
                    "\nQual modelo você pretende usar?\n"
                    "1 - OpenAI\n"
                    "2 - Google\n"
                    "\nEscolha uma opção: "
                ))
            break
        except ValueError:
            print("Informe apenas número")    
    

    pdf_ingestion_service = PdfIngestionService(envs=envs, modelo=modelo)
    search_service = SearchService(envs=envs, modelo=modelo)

    pdf_ingestion_service.ingest()

    main(search_service)

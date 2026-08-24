# Desafio MBA Engenharia de Software com IA — Ingestão e Busca

Aplicação de linha de comando que permite consultar o conteúdo de um PDF. O projeto resolve a busca semântica sobre o documento por meio de uma arquitetura RAG (*Retrieval-Augmented Generation*): o PDF é dividido em trechos, transformado em embeddings, armazenado no PostgreSQL com pgvector e recuperado para compor a resposta de um modelo de IA.

## Funcionalidades

- Ingestão de um arquivo PDF configurado pela variável `PDF_PATH`.
- Divisão do conteúdo em trechos de até 1.000 caracteres, com sobreposição de 150 caracteres.
- Geração de embeddings com OpenAI ou Google, conforme a opção selecionada no terminal.
- Persistência e busca por similaridade no PostgreSQL com pgvector.
- Recuperação dos três trechos mais semelhantes à pergunta.
- Chat interativo no terminal, com histórico mantido em memória durante a execução.
- Prompt que restringe as respostas ao contexto recuperado do documento.

## Pré-requisitos

- Python 3.10 ou superior (o código utiliza anotações de tipo com união por `|`).
- `pip` para instalar as dependências Python.
- Docker e Docker Compose com suporte ao comando `docker compose`.
- Uma chave de API da OpenAI para executar o fluxo OpenAI.
- Para o fluxo Google, uma chave de API do Google configurada para a biblioteca `langchain-google-genai`.
- O arquivo PDF a ser consultado. O repositório já inclui `document.pdf`.

## Configuração

### 1. Instalar dependências

Na raiz do repositório, crie e ative um ambiente virtual (opcional, mas recomendado) e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

O repositório fornece o arquivo `.env.example`. Copie-o para criar o arquivo `.env` na raiz do projeto:

```bash
cp .env.example .env
```

Preencha no `.env` as variáveis já fornecidas pelo modelo. Não use credenciais reais em arquivos versionados.

```env
# Chaves de API
OPENAI_API_KEY=<sua_chave_openai>
GOOGLE_API_KEY=<sua_chave_google>

# Banco de dados
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=<nome_da_colecao>

# Documento
PDF_PATH=./document.pdf

# Modelos de embeddings
OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
GOOGLE_EMBEDDING_MODEL='models/embedding-001'
```

`DATABASE_URL` corresponde ao serviço definido em `docker-compose.yml`. Os valores padrão dos modelos de embeddings já constam no `.env.example`.

Observação sobre o fluxo Google: `src/chat.py` valida e entrega apenas `OPENAI_API_KEY`, `DATABASE_URL`, `PG_VECTOR_COLLECTION_NAME`, `PDF_PATH` e `OPENAI_EMBEDDING_MODEL` ao serviço. Já `src/ingest.py` e `src/search.py` também esperam `GOOGLE_EMBEDDING_MODEL` quando a opção Google é escolhida. Portanto, na versão atual, o fluxo Google exige um ajuste em `load_envs()` para incluir essa variável; apenas defini-la no `.env` não é suficiente.

### 3. Inicializar infraestrutura

Inicie o PostgreSQL com pgvector e o serviço que habilita a extensão `vector`:

```bash
docker compose up -d
```

Esse comando cria o banco `rag`, expõe-o na porta `5432`, mantém os dados no volume `postgres_data` e executa `CREATE EXTENSION IF NOT EXISTS vector` após o banco ficar saudável.

Confirme que os serviços foram inicializados antes de executar a aplicação:

```bash
docker compose ps
```

## Uso

### Passo 1: Ingerir o PDF e iniciar o chat

Execute o ponto de entrada a partir da raiz do repositório:

```bash
python src/chat.py
```

Escolha `1` para OpenAI ou `2` para Google. Na versão atual, a opção OpenAI é o fluxo configurável sem alteração de código, desde que as variáveis obrigatórias estejam preenchidas.

Ao iniciar, a aplicação:

1. Carrega o PDF informado em `PDF_PATH`.
2. Divide o conteúdo em trechos e preserva os metadados disponíveis.
3. Gera embeddings e grava os trechos na coleção definida em `PG_VECTOR_COLLECTION_NAME`.
4. Abre o chat interativo no terminal.

### Passo 2: Fazer perguntas

No prompt `Pergunta:`, digite uma pergunta relacionada ao PDF. Digite `sair`, `exit` ou `quit` para encerrar.

```text
Qual modelo você pretende usar?
1 - OpenAI
2 - Google

Escolha uma opção: 1
Arquivo minha_colecao carregado com sucesso!
Pergunta: Qual é o tema principal do documento?

Assistente: [resposta baseada nos trechos recuperados]

Pergunta: sair

Encerrando chat.
```

Em cada pergunta, o sistema busca os três documentos mais similares, monta o contexto com o histórico da sessão atual e envia o prompt ao modelo selecionado. Caso a informação não esteja explicitamente no contexto recuperado, o prompt orienta o modelo a informar que não possui dados suficientes.

## Estrutura do Projeto

```text
.
├── src/
│   ├── chat.py             # Ponto de entrada e interface interativa no terminal
│   ├── ingest.py           # Leitura do PDF, divisão em trechos e indexação vetorial
│   └── search.py           # Busca por similaridade, prompt e chamada ao modelo de chat
├── docker-compose.yml      # PostgreSQL com pgvector e criação da extensão vector
├── document.pdf            # Documento-fonte incluído para ingestão
├── .env.example            # Modelo das variáveis necessárias para execução
├── requirements.txt        # Dependências Python fixadas
└── README.md               # Documentação do projeto
```

## Tecnologias Utilizadas

- **Python**: linguagem da aplicação de linha de comando.
- **LangChain**: carregamento, divisão de texto, abstrações de modelos e histórico de chat.
- **PyPDF / PyPDFLoader**: leitura do conteúdo do PDF.
- **OpenAI e langchain-openai**: modelo de chat e embeddings na opção OpenAI.
- **Google Generative AI e langchain-google-genai**: modelo de chat e embeddings na opção Google.
- **PostgreSQL 17 com pgvector**: persistência dos embeddings e busca por similaridade.
- **langchain-postgres**: integração entre LangChain e a coleção vetorial no PostgreSQL.
- **Docker Compose**: inicialização da infraestrutura local do banco.

## Troubleshooting

### Erro: `Environment variable <NOME> não foi definida`

- **Causa mais provável:** o arquivo `.env` está ausente, não está na raiz do projeto ou falta uma das variáveis validadas em `load_envs()`.
- **Como verificar:** confira a existência do arquivo e os nomes das variáveis, sem expor seus valores.
- **Como corrigir:** crie ou complete o `.env` com `OPENAI_API_KEY`, `DATABASE_URL`, `PG_VECTOR_COLLECTION_NAME`, `PDF_PATH` e `OPENAI_EMBEDDING_MODEL`. Atualmente essas variáveis são obrigatórias inclusive quando a opção Google é selecionada.

```bash
test -f .env && echo ".env encontrado"
```

### Erro: falha de conexão com o PostgreSQL ou com pgvector

- **Causa mais provável:** os contêineres não foram iniciados, ainda não estão saudáveis ou a `DATABASE_URL` não aponta para `localhost:5432` quando a aplicação é executada na máquina host.
- **Como verificar:** verifique o estado e os logs dos serviços.
- **Como corrigir:** inicie a infraestrutura e aguarde o `postgres` ficar saudável; confirme que o serviço `bootstrap_vector_ext` concluiu com sucesso.

```bash
docker compose ps
docker compose logs postgres
docker compose logs bootstrap_vector_ext
```

### Erro: PDF não encontrado ou falha ao carregar o documento

- **Causa mais provável:** `PDF_PATH` aponta para um arquivo inexistente ou para um caminho relativo diferente do diretório em que o comando foi executado.
- **Como verificar:** execute o comando abaixo na raiz do repositório.
- **Como corrigir:** ajuste `PDF_PATH` no `.env`; para o PDF incluído no projeto, use `./document.pdf` e execute `python src/chat.py` a partir da raiz.

```bash
ls -l ./document.pdf
```

### Erro: `GOOGLE_EMBEDDING_MODEL` sem valor definido ou falha na opção Google

- **Causa mais provável:** além da variável e da chave Google, há uma inconsistência na implementação atual: `load_envs()` não inclui `GOOGLE_EMBEDDING_MODEL` no dicionário entregue aos serviços.
- **Como verificar:** escolha a opção `2` ao iniciar o programa e observe a exceção durante a criação dos embeddings.
- **Como corrigir:** o desenvolvedor deve adicionar `GOOGLE_EMBEDDING_MODEL` a `envs` em `src/chat.py` e garantir que ela esteja definida no `.env`. Não há configuração somente por ambiente que contorne essa limitação do código atual.

### Erro: resultados repetidos após reiniciar a aplicação

- **Causa mais provável:** `PdfIngestionService.ingest()` é chamado a cada inicialização e usa os mesmos identificadores (`doc-0`, `doc-1`, ...) para adicionar os documentos; o projeto não possui uma rotina própria de limpeza ou reindexação.
- **Como verificar:** execute o fluxo mais de uma vez usando a mesma coleção e observe os resultados retornados.
- **Como corrigir:** defina uma estratégia de limpeza ou atualização da coleção antes de reexecutar a ingestão. Essa estratégia não está implementada no repositório e precisa ser definida pelo desenvolvedor.

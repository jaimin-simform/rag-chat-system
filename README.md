## Project Structure

rag_chat_system/
├── src/
│   ├── ui.py         # Streamlit UI for RAG chat system
│   ├── pydantic_ai.py # Pydantic-based AI agent logic with expert-level comments
│   ├── crawler.py    # Web crawler for content extraction with detailed comments
│   └── db.py         # Database initialization with ChromaDB and clear explanations
├── config/
│   └── .env          # Environment variables
├── tests/            # Unit and integration tests
├── docs/             # Documentation
├── README.md         # Project documentation
├── pyproject.toml    # Poetry dependency management
└── .gitignore        # Files to ignore in Git


# Example README.md Content

## Dynamic RAG Chat System

This project is a dynamic Retrieval-Augmented Generation (RAG) chat system built using Streamlit, OpenAI API, ChromaDB, and web crawling for knowledge enrichment.

### Features
1. **Dynamic Knowledge Base:** Crawl websites and build a knowledge base.
2. **Real-Time Chat:** Ask questions about processed content.
3. **Efficient Crawling:** Supports sitemap-based and single-page crawling.
4. **Embeddings and Storage:** Utilizes OpenAI embeddings and ChromaDB for vector search.

### Prerequisites
- Python 3.10+
- Poetry for dependency management

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag_chat_system.git
cd rag_chat_system
```

2. Set up the environment:
```bash
cp config/.env
```
Add your OpenAI API key and other environment variables in the `.env` file.

3. Install dependencies using Poetry:
```bash
poetry install
```

### Running the Application
Activate the virtual environment and start the Streamlit app:
```bash
poetry shell
streamlit run src/ui.py
```

### Environment Variables
Ensure the `.env` file includes:
```ini
OPENAI_API_KEY=your-api-key
```

### Directory Breakdown
- `src/`: Application source code with expert-level comments for clarity
- `config/`: Environment variables and configurations

### License
This project is licensed under the MIT License.
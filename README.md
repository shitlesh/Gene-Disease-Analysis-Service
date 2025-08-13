# Gene-Disease Analysis Tool

A web application for analyzing relationships between genes and diseases using AI-powered analysis.

## Features

- **Authentication**: Username + API key authentication (OpenAI/Claude)
- **Gene-Disease Analysis**: AI-powered analysis of gene-disease relationships
- **Real-time Updates**: Server-sent events for live analysis progress
- **Analysis History**: View and manage previous analyses
- **Modern UI**: React-based responsive interface

## Quick Start

### Using Docker (Recommended)

```bash
# Clone and start the application
git clone <repository-url>
cd gene-disease-analysis
docker-compose up --build
```

Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Manual Setup

#### Backend
```bash
cd backend
poetry install
poetry run uvicorn app.main_basic:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend
```bash
cd frontend
npm install
npm start
```

## Usage

1. **Login**: Enter your username and API key (OpenAI or Claude)
2. **Analyze**: Input a gene name and disease name
3. **Results**: View real-time analysis progress and results
4. **History**: Access previous analyses in your session

## API Keys

The application supports:
- **OpenAI**: API keys starting with `sk-`
- **Claude/Anthropic**: API keys starting with `sk-ant-` or `anthropic-`

## Architecture

- **Backend**: FastAPI with async support
- **Frontend**: React with Redux for state management
- **Storage**: In-memory (session-based)
- **AI Integration**: OpenAI GPT and Anthropic Claude APIs
- **External APIs**: NCBI for biological context

## Development

### Project Structure
```
gene-disease-analysis/
├── backend/           # FastAPI application
├── frontend/          # React application
└── docker-compose.yml # Container orchestration
```

### Key Technologies
- **Backend**: FastAPI, Python 3.11+, Poetry
- **Frontend**: React 18, Redux Toolkit, Webpack
- **APIs**: OpenAI GPT, Anthropic Claude, NCBI Entrez

## Health Checks

- Backend: `GET /health`
- Frontend: Available at root URL

## License

This project is for research and educational use only.

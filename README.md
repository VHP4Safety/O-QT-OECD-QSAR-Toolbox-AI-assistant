# QSAR Toolbox Assistant

A streamlined chemical analysis application that interfaces with the QSAR Toolbox API to provide chemical property analysis, profiling, and hazard assessment.

## Features

- Chemical search by name or SMILES notation
- Physicochemical properties calculation
- Experimental data retrieval
- Chemical profiling analysis
- Downloadable raw data in multiple formats (JSON, CSV)
- **Agentic AI-powered analysis reports:** Utilizes a multi-agent system (Specialists + Synthesizer) powered by LangChain and OpenAI for comprehensive, context-aware report generation. Specialist agents analyze different data aspects (physical properties, environmental fate, profiling, experimental data) in parallel, and a synthesizer agent combines these analyses into a final report.

## Architecture

The application follows a multi-stage process:

1.  **Data Collection:** Retrieves chemical data (properties, experimental, profiling) sequentially from the QSAR Toolbox API.
2.  **Parallel Agent Analysis:** Invokes multiple specialized LLM agents (via LangChain) concurrently using `asyncio`. Each agent analyzes a specific data subset (e.g., physical properties, environmental fate) based on the user's context.
3.  **Report Synthesis:** A final LLM agent synthesizes the outputs from the specialist agents into a single, comprehensive report.

This agentic approach allows for focused analysis by specialists and efficient parallel processing of LLM tasks.

## Project Structure

```
streamlined_qsar_app/
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── .env.example           # Example environment variables
├── components/            # UI Components
│   ├── search.py          # Search interface
│   └── results.py         # Results display and downloads
└── utils/                 # Utility modules
    ├── api_client.py      # API client utilities
    ├── data_formatter.py  # Data formatting utilities
    ├── llm_utils.py       # OpenAI integration utilities
    ├── qsar_api.py        # QSAR Toolbox API client
    └── report_generator.py # Report generation logic
```

## Prerequisites

- Python 3.8 or higher
- QSAR Toolbox with API running (local installation)
- OpenAI API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/streamlined_qsar_app.git
   cd streamlined_qsar_app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Copy the example .env file
   cp .env.example .env
   
   # Edit the .env file with your actual API keys
   # OPENAI_API_KEY=your_openai_api_key_here
   # QSAR_TOOLBOX_API_URL=http://localhost:5000/api
   ```

5. Ensure QSAR Toolbox is running and accessible on localhost:5000 (or update the URL in your .env file)

## Running the Application

1. With your virtual environment activated, start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and visit `http://localhost:8501`

## Usage

1. Select your search method (Chemical Name or SMILES)
2. Enter the chemical identifier
3. (Optional) Provide an analysis context for targeted reports
4. Click "Analyze Chemical"
5. View results in the interactive dashboard
6. Download raw data or analysis reports as needed

## Environment Variables

The application requires the following environment variables to be set in a `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key for report generation
- `QSAR_TOOLBOX_API_URL`: URL of the QSAR Toolbox API (default: http://localhost:5000/api)

## QSAR Toolbox Setup

The application requires the QSAR Toolbox to be installed and running with the API enabled:

1. Install the latest version of the OECD QSAR Toolbox from the official website
2. Ensure the API service is running (typically on port 5000)
3. Verify API connectivity before using the application

## Contributing

Contributions to improve the application are welcome:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some improvement'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

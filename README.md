# 🧪 OECD QSAR Toolbox Assistant

**Analyze chemicals, assess hazards, and get Read-Across recommendations using a powerful multi-agent AI system connected to the OECD QSAR Toolbox.**

This application uses LangChain and OpenAI agents to interpret data fetched from the OECD QSAR Toolbox API, providing comprehensive chemical analysis reports.

## Features

* Chemical search by name or SMILES notation.
* Retrieval of physicochemical properties, experimental data, and chemical profiling information via the QSAR Toolbox API.
* Downloadable raw data (JSON, CSV) and synthesized AI reports.
* **Advanced AI Analysis:** A team of specialized AI agents analyzes the data:
    * **Chemical Context:** Confirms the chemical's identity.
    * **Physical Properties:** Analyzes physical characteristics.
    * **Environmental Fate:** Evaluates environmental behavior.
    * **Profiling/Reactivity:** Assesses reactivity and toxicological alerts.
    * **Experimental Data:** Interprets experimental results.
    * **Read-Across:** Suggests analogue chemicals and strategies.
    * **Synthesis:** Combines all analyses into a final report.

---

## ⚠️ CRITICAL REQUIREMENT: The OECD QSAR Toolbox API

**This application CANNOT function without the official OECD QSAR Toolbox software.**

* **What it is:** The OECD QSAR Toolbox is a separate, comprehensive software package developed by the OECD. It contains the necessary databases, calculation engines, and profiling tools used by this assistant.
* **How this app uses it:** This application acts as a **client**. It connects to the **Web API** of your locally running QSAR Toolbox instance to request data (like properties, experimental results, profiling) for a specific chemical. The AI agents in this application then process and interpret the data received *from* the Toolbox API.
* **Where to get it:** You must download and install it directly from the OECD: [OECD QSAR Toolbox Download Page](https://qsartoolbox.org/download/)
* **Operating System:** The OECD QSAR Toolbox software **runs only on Microsoft Windows**. Therefore, you need a Windows environment to run the Toolbox and its API.
* **License:** You must agree to the OECD QSAR Toolbox End User License Agreement (EULA) when you install it.

**Why can't this be simpler (e.g., using Docker)?**

The QSAR Toolbox EULA **strictly prohibits** redistribution, copying (except for backup), or modification of the software (see Articles 2.2.b, 2.2.c, 2.2.f of the EULA). This means:

1.  **No Bundling:** We cannot legally include the QSAR Toolbox software within this application's code or installer.
2.  **No Docker Image:** We cannot create or distribute a Docker image containing the QSAR Toolbox software, as this would involve unauthorized copying and redistribution.

Therefore, **manual installation and API activation by each user on a Windows machine is the only compliant way** to use this assistant application.

---

## Installation and Usage Guide

Follow these steps carefully:

**Phase 1: Install and Configure the OECD QSAR Toolbox (on Windows)**

*(This only needs to be done once)*

1.  **Go to a Windows Machine:** You need access to a computer running Microsoft Windows.
2.  **Download:** Visit the [OECD QSAR Toolbox Download Page](https://qsartoolbox.org/download/) and download the latest version of the Toolbox.
3.  **Install:** Run the installer and follow the on-screen prompts. Make sure you accept the End User License Agreement (EULA).
4.  **Launch QSAR Toolbox:** Start the QSAR Toolbox application.
5.  **Enable the Web API:**
    * Inside the Toolbox, find the settings or options menu (often under "Tools", "Options", or similar).
    * Locate the "Web API" or "Web Services" settings.
    * **Check the box to enable the Web API.**
    * Note down the **Port number** it will use (the default is usually `5000`).
    * Ensure the API is accessible. It usually defaults to `localhost` (`127.0.0.1`), which is fine if you run the Streamlit Assistant on the *same* Windows machine.
6.  **Keep Toolbox Running:** The QSAR Toolbox application must be running with the API enabled whenever you want to use this Streamlit Assistant.
7.  **(Optional) Find Windows IP Address:** If you plan to run the Streamlit Assistant on a *different* computer (e.g., your Mac or Linux machine) than the Windows PC running the Toolbox:
    * Open Command Prompt on the Windows PC (search for `cmd`).
    * Type `ipconfig` and press Enter.
    * Look for the "IPv4 Address" under your active network connection (e.g., Wi-Fi or Ethernet). It will look something like `192.168.1.105`. Note this down.
    * You may also need to configure the Windows Firewall on that PC to allow incoming connections on the port noted in step 5 (e.g., 5000).

**Phase 2: Install and Configure the Streamlit Assistant (This Project)**

*(Can be done on Windows, macOS, or Linux)*

1.  **Clone Repository:**
    ```bash
    # Replace with the actual URL of your repository
    git clone [https://github.com/yourusername/streamlined_qsar_app.git](https://github.com/yourusername/streamlined_qsar_app.git)
    cd streamlined_qsar_app
    ```
2.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    # Activate:
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment Variables:**
    * Make a copy of the example file:
        ```bash
        cp .env.example .env
        ```
    * **Edit the `.env` file:**
        ```dotenv
        # Your OpenAI API Key for the AI agents
        OPENAI_API_KEY=sk-YourActualOpenAIKeyHere

        # *** THIS IS THE MOST IMPORTANT SETTING ***
        # Set the URL for YOUR running QSAR Toolbox API (from Phase 1)

        # Option 1: Running this Assistant on the SAME Windows PC as the Toolbox
        QSAR_TOOLBOX_API_URL=http://localhost:5000/api

        # Option 2: Running this Assistant on a DIFFERENT machine (Mac, Linux, another PC)
        # Use the Windows PC's IP address (from Phase 1, step 7) and the correct port (usually 5000)
        # Example: QSAR_TOOLBOX_API_URL=[http://192.168.1.105:5000/api](http://192.168.1.105:5000/api)
        ```
        *Uncomment the correct `QSAR_TOOLBOX_API_URL` line and replace the placeholder IP if necessary.*

**Phase 3: Run the Streamlit Assistant**

1.  **Ensure Toolbox is Running:** Double-check that the OECD QSAR Toolbox is running on the Windows machine and its Web API is active (Phase 1, step 6).
2.  **Activate Environment:** Make sure your `venv` virtual environment is activated (Phase 2, step 2).
3.  **Start the App:**
    ```bash
    streamlit run app.py
    ```
4.  **Access in Browser:** Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).
5.  **Check Connection:** The sidebar in the application should show "✅ Connected to QSAR Toolbox". If it shows an error, double-check the Toolbox is running, the API is enabled, and the `QSAR_TOOLBOX_API_URL` in your `.env` file is correct and reachable from the machine running the assistant.

## How to Use the Assistant

1.  Use the sidebar to choose search type (Name or SMILES) and enter the identifier.
2.  Optionally, provide context for the AI analysis (e.g., "assess potential for liver toxicity"). [cite: 446]
3.  Click "Analyze Chemical". [cite: 47]
4.  Wait for the analysis (progress bar will update). Data is fetched from the QSAR Toolbox API, then processed by the AI agents.
5.  Explore the results in the tabs: Chemical Overview, Properties, Experimental Data, Profiling.
6.  View the final synthesized report generated by the AI. [cite: 28]
7.  Download raw data or reports using the download buttons.

## Project Structure

```
streamlined_qsar_app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── .env.example           # Environment variable template
├── README.md              # This file
├── CONTRIBUTING.md        # Contribution guidelines
├── LICENSE                # MIT License
├── components/            # UI parts
│   ├── search.py
│   └── results.py
├── utils/                 # Backend logic
│   ├── qsar_api.py        # Client for the QSAR Toolbox Web API <--- Core Dependency
│   ├── llm_utils.py       # AI Agent logic (LangChain/OpenAI)
│   ├── data_formatter.py  # Data cleaning
│   └── prompts.yaml       # AI Agent prompts
└── tests/                 # Automated tests
    └── ...
```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines. Note that running tests related to `qsar_api.py` might require the QSAR Toolbox setup outlined above, although many tests use mocking.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. The OECD QSAR Toolbox has its own separate license agreement that you must adhere to.

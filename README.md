# LinkedIn Post Generator

A Gradio-based LinkedIn post generator that uses AI agents to research a topic, write a post, and validate the final result.

## Features

- Generates LinkedIn-ready posts from a topic, tone, and post type
- Uses research, writing, and validation agents
- Provides validation score, suggestions, and word count
- Includes both a Gradio web app and a CLI entry point

## Requirements

- Python 3.11.9
- API keys for:
  - `GROQ_API_KEY`
  - `TAVILY_API_KEY`

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Run the Gradio app

Start the web UI with:

```bash
python gradio_app.py
```

If you are deploying to a platform that provides a `PORT` environment variable, the app will use it automatically. Otherwise it defaults to `7860`.

## Run the CLI version

You can also run the command-line version directly:

```bash
python linkedin_generator.py
```

## Usage

In the Gradio app, enter:

- a topic
- a tone: `professional`, `casual`, or `thought-leader`
- a post type: `story`, `hot-take`, `announcement`, `lesson-learned`, or `thought-leader`

Then click **Generate Post** to create the LinkedIn post.

## Project Files

- `gradio_app.py` - web UI for generating posts
- `linkedin_generator.py` - core generator logic and agent workflow
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python runtime version for deployment

## Notes

- The generator uses Tavily for research and Groq-backed LLM calls through CrewAI.
- A valid topic is required before generation can begin.
- The app is configured to listen on `0.0.0.0` for deployment compatibility.

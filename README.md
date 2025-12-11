# Odyssey Python Client Example

Example project demonstrating how to use the [Odyssey Python client](https://github.com/odysseyml/odyssey-python).

## Setup

```bash
# Clone the repo
git clone https://github.com/odysseyml/odyssey-python-client-example.git
cd odyssey-python-client-example

# Install dependencies
uv sync
```

## Usage

```bash
# Set your API key
export ODYSSEY_API_KEY="ody_your_api_key_here"

# Run the example
uv run python main.py
```

### Custom prompts

```bash
uv run python main.py --prompt "A forest" --interaction "A deer walks by"
```

### Debug mode

```bash
uv run python main.py --debug
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `c` | Connect to Odyssey |
| `s` | Start stream |
| `i` | Send interaction |
| `e` | End stream |
| `d` | Disconnect |
| `p` | Toggle portrait/landscape |
| `q` | Quit |

## Requirements

- Python 3.12+
- OpenCV (installed automatically)

## Links

- [Odyssey Python Client](https://github.com/odysseyml/odyssey-python)
- [API Reference](https://github.com/odysseyml/odyssey-python/blob/main/API_REFERENCE.md)
- [Get an API key](https://documentation.api.odyssey.ml/)

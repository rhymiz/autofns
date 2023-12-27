# AutoFNS

## Introduction

AutoFNS is a wrapper around OpenAI's Chat Completion API. 

It allows you to define functions as you normally would, and have them be called
automatically when a message containing tool calls is detected. This allows you to focus on
more important things, like the logic of your application.

Note: 🧪 This is a work in progress and may not be ready for production use.

## Requirements

- python >= 3.10
- openai

## Installation

```bash
pip install autofns
```

## Usage

```python
from autofns import AutoFNS

FUNCTION_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_temp_units",
            "description": "Get a list of temperature units",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get the current temperature in a city",
        },
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get the temperature of",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to get the temperature in",
                    "enum": ["Fahrenheit", "Celsius", "Kelvin"],
                }
            },
            "required": ["city", "unit"],
        }
    },
]


def get_temp_units():
    return ["Fahrenheit", "Celsius", "Kelvin"]


def get_current_temperature(city: str, unit: str):
    return "The current temperature in {} is {} degrees {}".format(
        city, 72, unit
    )


fns = AutoFNS(
    "gpt-4-32k",
    fns_mapping={
        "get_temp_units": get_temp_units,
        "get_current_temperature": get_current_temperature
    },
    fns_definitions=FUNCTION_DEFINITIONS,
)

result = fns.create_completion(messages=[...])
```


You can also use the `AutoFNSAsync` class to use async functions:

```python
from autofns import AutoFNSAsync

fns = AutoFNSAsync(...)

result = await fns.create_completion(messages=[...])
```

## License
This project is licensed under the terms of the MIT license.

## Contributing
Contributions are welcome! Please open an issue or a pull request.
# Synced from the server; standalone demo blocks omitted.
TOOLS = [
    {
      "name": "search",
      "description": "Searches for information related to query and displays topn results.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "The search query string"
          },
          "topn": {
            "type": "integer",
            "description": "Number of top results to display",
            "default": 10
          },
          "source": {
            "type": "string",
            "description": "Source to search within",
            "enum": [
              "web",
              "news"
            ],
            "default": "web"
          }
        },
        "required": [
          "query"
        ],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema"
      }
    },
    {
      "name": "open",
      "description": "Opens the link id from the page indicated by cursor starting at line number loc, showing num_lines lines. Valid link ids are displayed with the formatting: 【{id}†.*】. If cursor is not provided, the most recent page is implied. If id is a string, it is treated as a fully qualified URL associated with source. If loc is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available. Use this function without id to scroll to a new location of an opened page.",
      "parameters": {
        "type": "object",
        "properties": {
          "id": {
            "description": "Link ID (number) or fully qualified URL (string)",
            "anyOf": [
              {
                "type": "integer"
              },
              {
                "type": "string"
              }
            ],
            "default": -1
          },
          "cursor": {
            "type": "integer",
            "description": "Page cursor indicator",
            "default": -1
          },
          "loc": {
            "type": "integer",
            "description": "Starting line number",
            "default": -1
          },
          "num_lines": {
            "type": "integer",
            "description": "Number of lines to show",
            "default": -1
          },
          "view_source": {
            "type": "boolean",
            "description": "Whether to view source",
            "default": False
          },
          "source": {
            "type": "string",
            "description": "Source associated with the URL",
            "enum": [
              "web",
              "news"
            ],
            "default": "web"
          }
        },
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema"
      }
    },
    {
      "name": "find",
      "description": "Finds exact matches of pattern in the current page, or the page given by cursor.",
      "parameters": {
        "type": "object",
        "properties": {
          "pattern": {
            "type": "string",
            "description": "The exact pattern to search for"
          },
          "cursor": {
            "type": "integer",
            "description": "Page cursor to search in",
            "default": -1
          }
        },
        "required": [
          "pattern"
        ],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema"
      }
    },
    {
      "name": "finish",
      "description": "Finish the task. Use this function when you have found the information you need.",
      "parameters": {
        "type": "object",
        "properties": {}
      }
    }
  ]

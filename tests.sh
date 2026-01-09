#!/bin/bash

MODULE=$1

uv run pytest src/tests/test_$MODULE.py